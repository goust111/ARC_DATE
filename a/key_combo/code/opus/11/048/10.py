from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
PIECE_COLOR = 5
MARKER_COLORS = (1, 2, 3, 4)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PROFILE_FAMILIES = [
    [
        (1, 1, 1, 1),
        (1, 2, 2, 1),
        (2, 1, 1, 2),
        (2, 2, 2, 2),
    ],
    [
        (1, 1, 1, 1, 1),
        (1, 1, 2, 1, 1),
        (1, 2, 1, 2, 1),
        (2, 1, 1, 1, 2),
    ],
]
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray | list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    arr = np.asarray(coords, dtype=int)
    r0 = int(arr[:, 0].min())
    c0 = int(arr[:, 1].min())
    r1 = int(arr[:, 0].max())
    c1 = int(arr[:, 1].max())
    return r0, c0, r1, c1


def _normalize(coords: np.ndarray | list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    arr = np.asarray(coords, dtype=int)
    r0, c0, _, _ = _bbox(arr)
    pts = {(int(r - r0), int(c - c0)) for r, c in arr}
    return tuple(sorted(pts))


def _coords_to_mask(coords: tuple[tuple[int, int], ...]) -> np.ndarray:
    arr = np.asarray(coords, dtype=int)
    h = int(arr[:, 0].max()) + 1
    w = int(arr[:, 1].max()) + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[arr[:, 0], arr[:, 1]] = True
    return mask


def _orientations(coords: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    base = _coords_to_mask(_normalize(coords))
    out: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for flip in (False, True):
        work = np.fliplr(base) if flip else base
        for k in range(4):
            rot = np.rot90(work, k)
            ori = tuple(sorted(map(tuple, np.argwhere(rot))))
            if ori not in seen:
                seen.add(ori)
                out.append(ori)
    return out


def _horizontal_profile(coords: tuple[tuple[int, int], ...]) -> tuple[int, ...] | None:
    pts = _normalize(coords)
    arr = np.asarray(pts, dtype=int)
    h = int(arr[:, 0].max()) + 1
    rows = [[] for _ in range(h)]
    for r, c in pts:
        rows[r].append(c)
    profile = []
    for cols in rows:
        cols = sorted(cols)
        if not cols:
            return None
        if cols[0] != 0 or cols != list(range(cols[-1] + 1)):
            return None
        profile.append(len(cols))
    return tuple(profile)


def _vertical_profile(coords: tuple[tuple[int, int], ...]) -> tuple[int, ...] | None:
    pts = _normalize(coords)
    arr = np.asarray(pts, dtype=int)
    w = int(arr[:, 1].max()) + 1
    cols = [[] for _ in range(w)]
    for r, c in pts:
        cols[c].append(r)
    profile = []
    for rows in cols:
        rows = sorted(rows)
        if not rows:
            return None
        if rows[0] != 0 or rows != list(range(rows[-1] + 1)):
            return None
        profile.append(len(rows))
    return tuple(profile)


def _valid_profiles_for_piece(coords: tuple[tuple[int, int], ...], axis: str) -> list[tuple[int, ...]]:
    profiles = set()
    for ori in _orientations(coords):
        prof = _horizontal_profile(ori) if axis == "h" else _vertical_profile(ori)
        if prof is not None:
            profiles.add(prof)
    return sorted(profiles)


def _valid_profile_combos(piece_coords: list[tuple[tuple[int, int], ...]], axis: str) -> list[tuple[tuple[int, ...], ...]]:
    options = [_valid_profiles_for_piece(piece, axis) for piece in piece_coords]
    if any(not opts for opts in options):
        return []
    combos = []
    for combo in itertools.product(*options):
        if len({len(profile) for profile in combo}) != 1:
            continue
        totals = [sum(v) for v in zip(*combo)]
        if len(set(totals)) == 1:
            combos.append(combo)
    return combos


def _render_profiles(profiles: tuple[tuple[int, ...], ...], colors: list[int], axis: str) -> np.ndarray:
    if axis == "h":
        h = len(profiles[0])
        w = sum(profile[0] for profile in profiles)
        out = np.zeros((h, w), dtype=int)
        for r in range(h):
            c = 0
            for profile, color in zip(profiles, colors):
                width = int(profile[r])
                out[r, c : c + width] = int(color)
                c += width
        return out

    w = len(profiles[0])
    h = sum(profile[0] for profile in profiles)
    out = np.zeros((h, w), dtype=int)
    for c in range(w):
        r = 0
        for profile, color in zip(profiles, colors):
            height = int(profile[c])
            out[r : r + height, c] = int(color)
            r += height
    return out


def _piece_distance(point: tuple[int, int], coords: np.ndarray) -> int:
    pr, pc = point
    return min(abs(int(pr) - int(r)) + abs(int(pc) - int(c)) for r, c in coords)


def _match_markers_to_pieces(grid: np.ndarray) -> tuple[list[tuple[int, int, int]], list[np.ndarray]]:
    pieces = _components_mask(grid == PIECE_COLOR)
    assert len(pieces) == 4, len(pieces)

    markers = []
    for color in MARKER_COLORS:
        coords = np.argwhere(grid == color)
        assert coords.shape == (1, 2), (color, coords.shape)
        markers.append((int(coords[0, 0]), int(coords[0, 1]), int(color)))

    matched = [None] * 4
    used_pieces = set()
    for marker in markers:
        point = (marker[0], marker[1])
        distances = [_piece_distance(point, piece) for piece in pieces]
        best = min(distances)
        nearest = [i for i, dist in enumerate(distances) if dist == best]
        assert len(nearest) == 1, (marker, distances)
        piece_idx = nearest[0]
        assert piece_idx not in used_pieces, (marker, piece_idx)
        used_pieces.add(piece_idx)
        matched[piece_idx] = marker

    assert len(used_pieces) == 4
    piece_markers = [marker for marker in matched if marker is not None]
    return piece_markers, pieces


def _detect_axis(markers: list[tuple[int, int, int]]) -> str:
    rows = [marker[0] for marker in markers]
    cols = [marker[1] for marker in markers]
    row_span = max(rows) - min(rows)
    col_span = max(cols) - min(cols)
    assert row_span != col_span, (row_span, col_span)
    return "v" if row_span > col_span else "h"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble the four grey jigsaw strips into a rectangle whose piece order is encoded by singleton marker positions, then recolor each strip by its matched marker.

    Input structure:
    - A black background contains exactly four grey 4-connected puzzle pieces and four singleton markers of distinct colors.
    - Each marker is uniquely closest in Manhattan distance to exactly one grey piece.
    - The four markers have a dominant spread axis: either their row span is larger or their column span is larger.

    Rule:
    - Match every marker to the uniquely nearest grey piece.
    - If marker row span is larger, sort markers top-to-bottom and place their matched pieces in four top-to-bottom slots; otherwise sort markers left-to-right and place their matched pieces in four left-to-right slots.
    - For each ordered piece, try all rotations and reflections and keep the unique combination whose four ordered pieces form a gapless rectangle.
    - Recolor each assembled piece with the color of its matched marker and output the tight assembled rectangle.

    Key insight:
    - The markers do two jobs at once: nearest-neighbor matching picks which marker belongs to which piece, and the markers' dominant-axis ordering encodes the assembly sequence of the final rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    piece_markers, pieces = _match_markers_to_pieces(g)
    axis = _detect_axis(piece_markers)
    axis_idx = 0 if axis == "v" else 1
    ordered = sorted(zip(piece_markers, pieces), key=lambda item: item[0][axis_idx])
    ordered_colors = [int(marker[2]) for marker, _piece in ordered]
    ordered_pieces = [_normalize(piece) for _marker, piece in ordered]

    combos = _valid_profile_combos(ordered_pieces, axis)
    assert len(combos) == 1, len(combos)
    return _render_profiles(combos[0], ordered_colors, axis)


def _profile_to_coords(profile: tuple[int, ...], axis: str) -> tuple[tuple[int, int], ...]:
    if axis == "h":
        return tuple((r, c) for r, width in enumerate(profile) for c in range(width))
    return tuple((r, c) for c, height in enumerate(profile) for r in range(height))


def _bbox_gap(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> int:
    vertical = max(0, max(b1[0] - b2[2] - 1, b2[0] - b1[2] - 1))
    horizontal = max(0, max(b1[1] - b2[3] - 1, b2[1] - b1[3] - 1))
    return vertical + horizontal


def _apply_offset(coords: tuple[tuple[int, int], ...], r0: int, c0: int) -> list[tuple[int, int]]:
    return [(int(r0 + r), int(c0 + c)) for r, c in coords]


def _center_row(bb: tuple[int, int, int, int]) -> int:
    return (int(bb[0]) + int(bb[2])) // 2


def _center_col(bb: tuple[int, int, int, int]) -> int:
    return (int(bb[1]) + int(bb[3])) // 2


def _place_horizontal(
    piece_shapes: list[tuple[tuple[int, int], ...]],
    h: int,
    w: int,
    rng: random.Random,
) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int]]]:
    row_slots = [1, 5, 9, 13]
    rng.shuffle(row_slots)
    col_anchors = [max(1, w // 2 - 2), max(1, w // 2 - 1), max(1, w // 2), max(1, w // 2 + 1)]
    rng.shuffle(col_anchors)

    placed = []
    bbs = []
    nontrivial = False
    for idx, base in enumerate(piece_shapes):
        options = _orientations(base)
        rng.shuffle(options)
        if not nontrivial:
            options.sort(key=lambda ori: ori != _normalize(base), reverse=True)
        chosen = None
        for ori in options:
            _, _, r1, c1 = _bbox(ori)
            height = r1 + 1
            width = c1 + 1
            r0 = row_slots[idx]
            if r0 + height > h - 1:
                continue
            c0 = min(max(1, col_anchors[idx]), w - width - 1)
            abscoords = _apply_offset(ori, r0, c0)
            bb = _bbox(abscoords)
            if any(_bbox_gap(bb, other) < 2 for other in bbs):
                continue
            chosen = abscoords
            bbs.append(bb)
            if ori != _normalize(base):
                nontrivial = True
            break
        assert chosen is not None
        placed.append(chosen)

    marker_templates = [
        [1, 4, w - 7, w - 2],
        [1, 5, w - 8, w - 3],
        [2, 6, w - 7, w - 2],
        [1, 6, w - 8, w - 2],
    ]
    rng.shuffle(marker_templates)
    for cols in marker_templates:
        if not (0 <= cols[0] < cols[1] < cols[2] < cols[3] < w):
            continue
        markers = []
        distances = []
        ok = True
        for piece, bb, col in zip(placed, bbs, cols):
            marker = (_center_row(bb), int(col))
            if marker in piece:
                ok = False
                break
            dists = [_piece_distance(marker, np.asarray(other, dtype=int)) for other in placed]
            own = _piece_distance(marker, np.asarray(piece, dtype=int))
            if own != min(dists) or dists.count(own) != 1:
                ok = False
                break
            markers.append(marker)
            distances.append(own)
        if ok and len(set(distances)) == 4:
            rows = [r for r, _c in markers]
            cols2 = [c for _r, c in markers]
            if max(cols2) - min(cols2) > max(rows) - min(rows):
                return placed, markers
    raise AssertionError("failed horizontal marker placement")


def _place_vertical(
    piece_shapes: list[tuple[tuple[int, int], ...]],
    h: int,
    w: int,
    rng: random.Random,
) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int]]]:
    col_slots = [1, 5, 9, 13]
    rng.shuffle(col_slots)
    row_anchors = [max(1, h // 2 - 2), max(1, h // 2 - 1), max(1, h // 2), max(1, h // 2 + 1)]
    rng.shuffle(row_anchors)

    placed = []
    bbs = []
    nontrivial = False
    for idx, base in enumerate(piece_shapes):
        options = _orientations(base)
        rng.shuffle(options)
        if not nontrivial:
            options.sort(key=lambda ori: ori != _normalize(base), reverse=True)
        chosen = None
        for ori in options:
            _, _, r1, c1 = _bbox(ori)
            height = r1 + 1
            width = c1 + 1
            c0 = col_slots[idx]
            if c0 + width > w - 1:
                continue
            r0 = min(max(1, row_anchors[idx]), h - height - 1)
            abscoords = _apply_offset(ori, r0, c0)
            bb = _bbox(abscoords)
            if any(_bbox_gap(bb, other) < 2 for other in bbs):
                continue
            chosen = abscoords
            bbs.append(bb)
            if ori != _normalize(base):
                nontrivial = True
            break
        assert chosen is not None
        placed.append(chosen)

    marker_templates = [
        [1, 4, h - 7, h - 2],
        [1, 5, h - 8, h - 3],
        [2, 6, h - 7, h - 2],
        [1, 6, h - 8, h - 2],
    ]
    rng.shuffle(marker_templates)
    for rows in marker_templates:
        if not (0 <= rows[0] < rows[1] < rows[2] < rows[3] < h):
            continue
        markers = []
        distances = []
        ok = True
        for piece, bb, row in zip(placed, bbs, rows):
            marker = (int(row), _center_col(bb))
            if marker in piece:
                ok = False
                break
            dists = [_piece_distance(marker, np.asarray(other, dtype=int)) for other in placed]
            own = _piece_distance(marker, np.asarray(piece, dtype=int))
            if own != min(dists) or dists.count(own) != 1:
                ok = False
                break
            markers.append(marker)
            distances.append(own)
        if ok and len(set(distances)) == 4:
            rows2 = [r for r, _c in markers]
            cols = [c for _r, c in markers]
            if max(rows2) - min(rows2) > max(cols) - min(cols):
                return placed, markers
    raise AssertionError("failed vertical marker placement")


def _paint_grid(h: int, w: int, pieces: list[list[tuple[int, int]]], markers: list[tuple[int, int]], colors: list[int]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for piece in pieces:
        for r, c in piece:
            grid[r, c] = PIECE_COLOR
    for (r, c), color in zip(markers, colors):
        assert grid[r, c] == BG
        grid[r, c] = int(color)
    return grid


def _check_sample(grid: np.ndarray, expected: np.ndarray, axis: str, piece_shapes: list[tuple[tuple[int, int], ...]]) -> None:
    out = main(grid)
    assert np.array_equal(out, expected)
    assert len(_components_mask(grid == PIECE_COLOR)) == 4

    markers = [(int(coords[0, 0]), int(coords[0, 1]), color) for color in MARKER_COLORS for coords in [np.argwhere(grid == color)]]
    rows = [marker[0] for marker in markers]
    cols = [marker[1] for marker in markers]
    row_span = max(rows) - min(rows)
    col_span = max(cols) - min(cols)
    assert row_span != col_span
    assert (axis == "h" and col_span > row_span) or (axis == "v" and row_span > col_span)

    piece_markers, pieces = _match_markers_to_pieces(grid)
    distances = [_piece_distance((marker[0], marker[1]), piece) for marker, piece in zip(piece_markers, pieces)]
    assert len(set(distances)) == 4

    combos = _valid_profile_combos(piece_shapes, axis)
    assert len(combos) == 1
    wrong = _render_profiles(combos[0], list(reversed(MARKER_COLORS)), axis)
    assert not np.array_equal(wrong, expected)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    axis = rng.choice(["h", "v"])
    family = [tuple(profile) for profile in rng.choice(PROFILE_FAMILIES)]
    rng.shuffle(family)
    colors = rng.sample(list(MARKER_COLORS), 4)
    expected = _render_profiles(tuple(family), colors, axis)

    piece_shapes = [_profile_to_coords(profile, axis) for profile in family]
    assert len(_valid_profile_combos(piece_shapes, axis)) == 1

    h = rng.randint(19, 22)
    w = rng.randint(19, 22)
    try:
        if axis == "h":
            pieces, markers = _place_horizontal(piece_shapes, h, w, rng)
        else:
            pieces, markers = _place_vertical(piece_shapes, h, w, rng)
    except AssertionError:
        return None

    grid = _paint_grid(h, w, pieces, markers, colors)
    try:
        _check_sample(grid, expected, axis, piece_shapes)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
