from __future__ import annotations

import random
from itertools import product

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))

# Generator configuration.
PIECE_COUNT_OPTIONS = (4, 6, 8)
HEIGHT_RANGE = (4, 6)
BODY_WIDTH_RANGE = (3, 7)
GRID_SIDE_MIN = 18
GRID_SIDE_MAX = 30
PLACEMENT_MARGIN = 1
MAX_GENERATION_ATTEMPTS = 400
MAX_PLACEMENT_ATTEMPTS = 250


def _row_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return the contiguous filled span of each row in a monochrome piece mask."""
    spans: list[tuple[int, int]] = []
    for row in mask:
        cols = np.flatnonzero(row)
        assert cols.size > 0, "every piece row must contain foreground pixels"
        c0 = int(cols[0])
        c1 = int(cols[-1])
        assert np.all(row[c0 : c1 + 1]), "each piece row must be a single run"
        spans.append((c0, c1))
    return spans


def _extract_pieces(grid: np.ndarray) -> list[dict]:
    """Extract one monochrome piece per non-black color."""
    pieces: list[dict] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != BG):
        coords = np.argwhere(grid == color)
        assert coords.size > 0, "piece color must occur"
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        mask = grid[r0 : r1 + 1, c0 : c1 + 1] == color
        spans = _row_spans(mask)
        pieces.append(
            {
                "color": int(color),
                "bbox": (r0, c0, r1, c1),
                "mask": mask,
                "spans": spans,
            }
        )
    assert pieces, "expected at least one puzzle piece"
    return pieces


def _decode_piece_profiles(pieces: list[dict]) -> tuple[int, int]:
    """Infer shared piece height and base body width from all extracted pieces."""
    heights = {int(piece["mask"].shape[0]) for piece in pieces}
    assert len(heights) == 1, "all pieces must share one height"
    height = int(next(iter(heights)))

    body_width = max(c1 for piece in pieces for _c0, c1 in piece["spans"])
    assert body_width >= 3, "piece body width must be at least 3"
    return height, int(body_width)


def _annotate_piece(piece: dict, body_width: int) -> dict:
    left = [int(c0) for c0, _c1 in piece["spans"]]
    right = [int(c1 - (body_width - 1)) for _c0, c1 in piece["spans"]]

    assert all(v in (0, 1) for v in left), "left edge must be flat or indented by one cell"
    assert all(v in (0, 1) for v in right), "right edge must be flat or protrude by one cell"
    assert min(left) == 0, "each generated piece must expose its left reference column"

    out = dict(piece)
    out["left"] = tuple(left)
    out["right"] = tuple(right)
    return out


def _order_chain(pieces: list[dict]) -> list[dict]:
    start_candidates = [piece for piece in pieces if all(v == 0 for v in piece["left"])]
    end_candidates = [piece for piece in pieces if all(v == 0 for v in piece["right"])]
    assert len(start_candidates) == 1, "expected exactly one flat-left start piece"
    assert len(end_candidates) == 1, "expected exactly one flat-right end piece"

    start = start_candidates[0]
    end = end_candidates[0]

    left_lookup: dict[tuple[int, ...], list[dict]] = {}
    for piece in pieces:
        left_lookup.setdefault(piece["left"], []).append(piece)

    chain = [start]
    used = {int(start["color"])}
    while chain[-1] is not end:
        key = chain[-1]["right"]
        matches = [
            piece
            for piece in left_lookup.get(key, [])
            if int(piece["color"]) not in used
        ]
        assert len(matches) == 1, "every right edge must match one unique next piece"
        nxt = matches[0]
        chain.append(nxt)
        used.add(int(nxt["color"]))

    assert len(chain) == len(pieces), "the chain must use every piece exactly once"
    return chain


def _assemble(chain: list[dict], height: int, body_width: int) -> np.ndarray:
    out = np.full((height, body_width * len(chain)), BG, dtype=int)
    for i, piece in enumerate(chain):
        base = i * body_width
        color = int(piece["color"])
        for row, (left, right) in enumerate(zip(piece["left"], piece["right"], strict=False)):
            start = base + int(left)
            stop = base + body_width + int(right)
            out[row, start:stop] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered puzzle fragments into one horizontal strip by matching
    each piece's right-edge tooth pattern to the unique complementary left-edge pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    raw_pieces = _extract_pieces(g)
    height, body_width = _decode_piece_profiles(raw_pieces)
    pieces = [_annotate_piece(piece, body_width) for piece in raw_pieces]
    chain = _order_chain(pieces)
    return _assemble(chain, height, body_width)


def _all_profiles(height: int) -> list[tuple[int, ...]]:
    """Profiles usable on an internal seam: mixed rows only, so the edge is visible."""
    profiles = []
    for bits in product((0, 1), repeat=height):
        if 0 < sum(bits) < height:
            profiles.append(tuple(int(v) for v in bits))
    return profiles


def _render_piece(
    color: int,
    left_profile: tuple[int, ...],
    right_profile: tuple[int, ...],
    body_width: int,
) -> np.ndarray:
    width = body_width + int(max(right_profile))
    patch = np.full((len(left_profile), width), BG, dtype=int)
    for row, (left, right) in enumerate(zip(left_profile, right_profile, strict=False)):
        patch[row, int(left) : body_width + int(right)] = int(color)
    return patch


def _expected_output(
    chain_profiles: list[tuple[tuple[int, ...], tuple[int, ...], int]],
    body_width: int,
) -> np.ndarray:
    height = len(chain_profiles[0][0])
    out = np.full((height, body_width * len(chain_profiles)), BG, dtype=int)
    for i, (left_profile, right_profile, color) in enumerate(chain_profiles):
        base = i * body_width
        for row, (left, right) in enumerate(zip(left_profile, right_profile, strict=False)):
            out[row, base + int(left) : base + body_width + int(right)] = int(color)
    return out


def _nontrivial_positioning(placements: list[tuple[int, int, int]]) -> bool:
    """Reject layouts where sorting by one coordinate accidentally reveals the chain."""
    chain_order = [idx for idx, _r0, _c0 in placements]
    by_col = [idx for idx, _r0, _c0 in sorted(placements, key=lambda x: (x[2], x[1]))]
    by_row = [idx for idx, _r0, _c0 in sorted(placements, key=lambda x: (x[1], x[2]))]
    target = list(range(len(placements)))
    reverse = list(reversed(target))
    return by_col not in (target, reverse) and by_row not in (target, reverse)


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    H, W = occupied.shape
    rr0 = max(0, r0 - PLACEMENT_MARGIN)
    cc0 = max(0, c0 - PLACEMENT_MARGIN)
    rr1 = min(H, r0 + h + PLACEMENT_MARGIN)
    cc1 = min(W, c0 + w + PLACEMENT_MARGIN)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _place_patches(patches: list[np.ndarray]) -> tuple[np.ndarray, list[tuple[int, int, int]]] | None:
    piece_area = sum(int(np.count_nonzero(patch)) for patch in patches)
    max_h = max(int(patch.shape[0]) for patch in patches)
    max_w = max(int(patch.shape[1]) for patch in patches)
    grid_choices = [
        (H, W)
        for H in range(GRID_SIDE_MIN, GRID_SIDE_MAX + 1)
        for W in range(GRID_SIDE_MIN, GRID_SIDE_MAX + 1)
        if H >= max_h and W >= max_w and H * W >= piece_area + 40
    ]
    random.shuffle(grid_choices)

    for H, W in grid_choices:
        occupied = np.zeros((H, W), dtype=bool)
        grid = np.full((H, W), BG, dtype=int)
        placements: list[tuple[int, int, int]] = []

        placement_order = list(range(len(patches)))
        random.shuffle(placement_order)

        ok = True
        for idx in placement_order:
            patch = patches[idx]
            ph, pw = map(int, patch.shape)
            placed = False
            for _ in range(MAX_PLACEMENT_ATTEMPTS):
                r0 = random.randint(0, H - ph)
                c0 = random.randint(0, W - pw)
                if not _can_place(occupied, r0, c0, ph, pw):
                    continue
                mask = patch != BG
                grid[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]
                occupied[r0 : r0 + ph, c0 : c0 + pw] |= mask
                placements.append((idx, r0, c0))
                placed = True
                break
            if not placed:
                ok = False
                break

        if ok and _nontrivial_positioning(placements):
            return grid, placements
    return None


def _generate_profiles() -> tuple[list[tuple[tuple[int, ...], tuple[int, ...], int]], int]:
    n_pieces = random.choice(PIECE_COUNT_OPTIONS)

    valid_heights = [
        height
        for height in range(HEIGHT_RANGE[0], HEIGHT_RANGE[1] + 1)
        if len(_all_profiles(height)) >= n_pieces - 1
    ]
    height = random.choice(valid_heights)
    min_body_width = BODY_WIDTH_RANGE[0]
    max_body_width = min(BODY_WIDTH_RANGE[1], 30 // n_pieces)
    assert min_body_width <= max_body_width, "piece count must admit an ARC-sized output"
    body_width = random.randint(min_body_width, max_body_width)

    seam_profiles = random.sample(_all_profiles(height), n_pieces - 1)
    colors = random.sample(COLORS, n_pieces)

    chain_profiles: list[tuple[tuple[int, ...], tuple[int, ...], int]] = []
    for i in range(n_pieces):
        left_profile = tuple(0 for _ in range(height)) if i == 0 else seam_profiles[i - 1]
        right_profile = tuple(0 for _ in range(height)) if i == n_pieces - 1 else seam_profiles[i]
        chain_profiles.append((left_profile, right_profile, int(colors[i])))
    return chain_profiles, body_width


def _attempt_generate() -> np.ndarray | None:
    chain_profiles, body_width = _generate_profiles()
    patches = [
        _render_piece(color, left_profile, right_profile, body_width)
        for left_profile, right_profile, color in chain_profiles
    ]
    placed = _place_patches(patches)
    if placed is None:
        return None

    grid, _placements = placed
    expected = _expected_output(chain_profiles, body_width)

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
