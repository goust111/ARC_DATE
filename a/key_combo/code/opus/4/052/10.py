from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
START = 1
WILDCARD = 7
MARKERS = (2, 3, 4, 5, 6, 8, 9)


def _bbox_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_pieces(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    pieces: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == BG:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) != BG:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            r0, c0, r1, c1 = _bbox_coords(coords)
            patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
            ph, pw = patch.shape
            assert len(coords) == ph * pw, "each fragment must be a solid rectangle"
            assert np.all(patch != BG), "rectangles must be fully occupied"
            pieces.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "patch": patch,
                    "top": r0,
                    "left": c0,
                    "h": ph,
                    "w": pw,
                }
            )

    assert 4 <= len(pieces) <= 6, "expected 4 to 6 rectangular fragments"
    return pieces


def _edge_compatible(left_patch: np.ndarray, right_patch: np.ndarray) -> bool:
    assert left_patch.shape[0] == right_patch.shape[0]
    a = left_patch[:, -1]
    b = right_patch[:, 0]
    mask = (a != WILDCARD) & (b != WILDCARD)
    return bool(np.all(a[mask] == b[mask]))


def _find_chain(pieces: list[dict]) -> list[dict]:
    starts = [piece for piece in pieces if int(piece["patch"][0, 0]) == START]
    assert len(starts) == 1, "expected exactly one blue-marked start fragment"

    used_ids = {id(starts[0])}
    chain = [starts[0]]
    current = starts[0]

    while True:
        target_left = int(current["left"]) + int(current["h"])
        candidates = [
            piece
            for piece in pieces
            if id(piece) not in used_ids
            and int(piece["left"]) == target_left
            and int(piece["h"]) == int(current["h"])
        ]
        matches = [piece for piece in candidates if _edge_compatible(current["patch"], piece["patch"])]
        if not matches:
            break
        assert len(matches) == 1, "chain successor must be unique after wildcard edge matching"
        current = matches[0]
        chain.append(current)
        used_ids.add(id(current))

    assert len(chain) >= 2, "chain must contain at least two fragments"
    return chain


def _assemble(chain: list[dict]) -> np.ndarray:
    height = int(chain[0]["h"])
    assert all(int(piece["h"]) == height for piece in chain)
    for left_piece, right_piece in zip(chain, chain[1:]):
        assert _edge_compatible(left_piece["patch"], right_piece["patch"])
    return np.concatenate([piece["patch"] for piece in chain], axis=1)


def main(grid: np.ndarray) -> np.ndarray:
    """Assemble only the chain-linked rectangular fragments into one horizontal strip.

    Input structure:
    - Black background surrounds 4 to 6 disconnected 4-connected rectangular fragments.
    - Each fragment is a fully occupied small rectangle made mostly of orange cells, with a few non-orange markers.
    - Exactly one fragment has a blue top-left cell, marking the chain start.

    Rule:
    - Extract every non-black connected component and treat its bbox as one fragment.
    - Start from the blue-marked fragment. A fragment can follow the current one only if its top-left column equals the current fragment's top-left column plus the current fragment height.
    - Among those column-valid candidates, keep the unique one whose left edge wildcard-matches the current fragment's right edge: orange matches anything, but whenever both touching edge cells are non-orange they must have the same color.
    - Repeat until no valid successor remains, discard every fragment outside the chain, and concatenate the chain fragments left-to-right with their top rows aligned.

    Key insight:
    - The spatial chain rule chooses which fragments are even eligible next, while the orange wildcard edge check removes same-column decoys and fixes the only valid assembly order.
    """
    pieces = _extract_pieces(grid)
    chain = _find_chain(pieces)
    return _assemble(chain)


def _random_signature(height: int) -> np.ndarray:
    sig = np.full(height, WILDCARD, dtype=int)
    rows = random.sample(range(height), k=random.randint(1, min(2, height)))
    for row in rows:
        sig[row] = random.choice(MARKERS)
    return sig


def _random_optional_edge(height: int) -> np.ndarray:
    edge = np.full(height, WILDCARD, dtype=int)
    for row in range(height):
        if random.random() < 0.25:
            edge[row] = random.choice(MARKERS)
    return edge


def _decorate_piece(
    height: int,
    width: int,
    left_edge: np.ndarray | None,
    right_edge: np.ndarray | None,
    *,
    is_start: bool,
) -> np.ndarray:
    patch = np.full((height, width), WILDCARD, dtype=int)
    if left_edge is not None:
        patch[:, 0] = left_edge
    if right_edge is not None:
        patch[:, -1] = right_edge

    cells = [(r, c) for r in range(height) for c in range(width)]
    random.shuffle(cells)
    extra = random.randint(1, max(1, min(4, height * width // 2)))
    painted = 0
    for r, c in cells:
        if is_start and (r, c) == (0, 0):
            continue
        if c == 0 and left_edge is not None:
            continue
        if c == width - 1 and right_edge is not None:
            continue
        if patch[r, c] != WILDCARD:
            continue
        if random.random() < 0.55:
            patch[r, c] = random.choice(MARKERS)
            painted += 1
            if painted >= extra:
                break

    if is_start:
        patch[0, 0] = START
    return patch


def _boxes_clear(placed: list[tuple[int, int, int, int]], r: int, c: int, h: int, w: int, pad: int = 1) -> bool:
    r0 = r - pad
    c0 = c - pad
    r1 = r + h - 1 + pad
    c1 = c + w - 1 + pad
    for pr, pc, ph, pw in placed:
        q0 = pr - pad
        s0 = pc - pad
        q1 = pr + ph - 1 + pad
        s1 = pc + pw - 1 + pad
        if not (r1 < q0 or q1 < r0 or c1 < s0 or s1 < c0):
            return False
    return True


def _place_fixed_column(
    board_h: int,
    placed: list[tuple[int, int, int, int]],
    col: int,
    height: int,
    width: int,
) -> int | None:
    rows = list(range(board_h - height + 1))
    random.shuffle(rows)
    for row in rows:
        if _boxes_clear(placed, row, col, height, width):
            return row
    return None


def _place_anywhere(
    board_h: int,
    board_w: int,
    placed: list[tuple[int, int, int, int]],
    height: int,
    width: int,
    forbidden_cols: set[int],
) -> tuple[int, int] | None:
    candidates = [
        (r, c)
        for r in range(board_h - height + 1)
        for c in range(board_w - width + 1)
        if c not in forbidden_cols
    ]
    random.shuffle(candidates)
    for row, col in candidates:
        if _boxes_clear(placed, row, col, height, width):
            return row, col
    return None


def _expected_from_chain(chain_patches: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(chain_patches, axis=1)


def _validate_semantics(
    grid: np.ndarray,
    expected: np.ndarray,
    chain_positions: list[tuple[int, int]],
    chain_patches: list[np.ndarray],
    decoy_col: int,
    decoy_patch: np.ndarray,
    decoy_step: int,
) -> bool:
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False

    pieces = _extract_pieces(grid)
    chain = _find_chain(pieces)
    actual = _assemble(chain)
    if not np.array_equal(actual, expected):
        return False

    starts = [piece for piece in pieces if int(piece["patch"][0, 0]) == START]
    if len(starts) != 1:
        return False

    if len({r for r, _ in chain_positions}) == 1:
        return False

    for idx, piece in enumerate(chain):
        if not np.array_equal(piece["patch"], chain_patches[idx]):
            return False

    prev_patch = chain_patches[decoy_step]
    if int(chain_positions[decoy_step][1]) + prev_patch.shape[0] != decoy_col:
        return False
    if _edge_compatible(prev_patch, decoy_patch):
        return False

    target_matches = [
        piece
        for piece in pieces
        if int(piece["left"]) == decoy_col and int(piece["h"]) == int(prev_patch.shape[0])
    ]
    compatible_count = sum(_edge_compatible(prev_patch, piece["patch"]) for piece in target_matches)
    if compatible_count != 1:
        return False

    if not np.array_equal(main(grid), expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    chain_len = random.randint(3, 5)
    height = random.randint(2, 5)
    widths = [random.randint(2, 4) for _ in range(chain_len)]

    start_col = random.randint(0, 4)
    chain_cols = [start_col + i * height for i in range(chain_len)]

    left_edges = [np.full(height, WILDCARD, dtype=int) for _ in range(chain_len)]
    right_edges = [np.full(height, WILDCARD, dtype=int) for _ in range(chain_len)]
    interfaces = [_random_signature(height) for _ in range(chain_len - 1)]
    for idx, sig in enumerate(interfaces):
        right_edges[idx] = sig.copy()
        left_edges[idx + 1] = sig.copy()
    left_edges[0] = _random_optional_edge(height)
    right_edges[-1] = _random_optional_edge(height)

    chain_patches = [
        _decorate_piece(height, width, left_edges[idx], right_edges[idx], is_start=(idx == 0))
        for idx, width in enumerate(widths)
    ]
    expected = _expected_from_chain(chain_patches)
    if max(expected.shape) > 30:
        return None

    decoy_step = random.randint(0, chain_len - 2)
    mismatch_rows = [row for row in range(height) if int(interfaces[decoy_step][row]) != WILDCARD]
    if not mismatch_rows:
        return None
    wrong_left = left_edges[decoy_step + 1].copy()
    bad_row = random.choice(mismatch_rows)
    bad_color_choices = [color for color in MARKERS if color != int(chain_patches[decoy_step][:, -1][bad_row])]
    wrong_left[bad_row] = random.choice(bad_color_choices)
    decoy_width = random.randint(2, 4)
    decoy_patch = _decorate_piece(height, decoy_width, wrong_left, _random_optional_edge(height), is_start=False)
    decoy_col = chain_cols[decoy_step] + height

    extra_count = 1 if chain_len == 5 else random.randint(1, 2)
    total_pieces = chain_len + extra_count
    if total_pieces > 6:
        return None

    distractor_specs: list[tuple[np.ndarray, int | None]] = [(decoy_patch, decoy_col)]
    for _ in range(extra_count - 1):
        dh = random.randint(2, 5)
        dw = random.randint(2, 4)
        patch = _decorate_piece(dh, dw, _random_optional_edge(dh), _random_optional_edge(dh), is_start=False)
        distractor_specs.append((patch, None))

    min_w = max(
        chain_cols[-1] + widths[-1] + 1,
        decoy_col + decoy_width + 1,
        expected.shape[1] + 2,
        14,
    )
    if min_w > 30:
        return None
    board_w = random.randint(min_w, 30)

    min_h = max(height + 4, 12)
    board_h = random.randint(min_h, min(30, max(min_h, 18 + (height // 2))))

    placed: list[tuple[int, int, int, int]] = []
    chain_positions: list[tuple[int, int]] = []
    for col, patch in zip(chain_cols, chain_patches):
        row = _place_fixed_column(board_h, placed, col, patch.shape[0], patch.shape[1])
        if row is None:
            return None
        placed.append((row, col, patch.shape[0], patch.shape[1]))
        chain_positions.append((row, col))

    distractor_positions: list[tuple[int, int]] = []
    forbidden_cols = set(chain_cols) | {chain_cols[idx] + height for idx in range(chain_len - 1)}
    for patch, fixed_col in distractor_specs:
        if fixed_col is not None:
            row = _place_fixed_column(board_h, placed, fixed_col, patch.shape[0], patch.shape[1])
            if row is None:
                return None
            col = fixed_col
        else:
            pos = _place_anywhere(board_h, board_w, placed, patch.shape[0], patch.shape[1], forbidden_cols)
            if pos is None:
                return None
            row, col = pos
        placed.append((row, col, patch.shape[0], patch.shape[1]))
        distractor_positions.append((row, col))

    grid = np.zeros((board_h, board_w), dtype=int)
    for (row, col), patch in zip(chain_positions, chain_patches):
        grid[row : row + patch.shape[0], col : col + patch.shape[1]] = patch
    for (row, col), (patch, _fixed_col) in zip(distractor_positions, distractor_specs):
        grid[row : row + patch.shape[0], col : col + patch.shape[1]] = patch

    if not _validate_semantics(
        grid,
        expected,
        chain_positions,
        chain_patches,
        decoy_col,
        decoy_patch,
        decoy_step,
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chained-edge-matching input")
