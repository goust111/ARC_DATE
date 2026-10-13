from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _UTILS_DIR.exists() and str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import find_components, paste, random_free_location


BG = 0
GRAY = 5
PIECE_COLORS = (1, 2, 3, 4, 6)
OUTPUT_WIDTH = 7
TARGET_HEIGHTS = (3, 4)
NOISE_COUNT_BY_HEIGHT = {3: 2, 4: 3}
MAX_GENERATION_ATTEMPTS = 200


def _factor_pairs(area: int) -> list[tuple[int, int]]:
    out = []
    for h in range(2, int(area**0.5) + 1):
        if area % h == 0:
            w = area // h
            if h <= w and w > 1:
                out.append((h, w))
    return out


def _piece_signature(coords: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0 = np.min(coords, axis=0)
    shifted = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
    h = max(r for r, _ in shifted) + 1
    w = max(c for _, c in shifted) + 1
    return h, w, shifted


def _extract_pieces(
    grid: np.ndarray,
    *,
    keep_singletons: bool = False,
    drop_gray: bool = True,
) -> list[dict]:
    comps = find_components(
        np.asarray(grid, dtype=int),
        bg=BG,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    out = []
    for comp in comps:
        color = int(comp["color"])
        area = int(comp["area"])
        if drop_gray and color == GRAY:
            continue
        if not keep_singletons and area == 1:
            continue
        out.append(
            {
                "color": color,
                "area": area,
                "coords": np.asarray(comp["coords"], dtype=int),
            }
        )
    return out


def _pieces_to_types(pieces: list[dict]) -> list[dict]:
    counts = Counter()
    for piece in pieces:
        counts[(int(piece["color"]), _piece_signature(piece["coords"]))] += 1

    types = []
    for (color, (h, w, coords)), count in sorted(
        counts.items(),
        key=lambda item: (-len(item[0][1][2]), -item[0][1][0], -item[0][1][1], item[0][0]),
    ):
        types.append(
            {
                "color": int(color),
                "coords": tuple(coords),
                "h": int(h),
                "w": int(w),
                "count": int(count),
            }
        )
    return types


def _search_assemblies_for_dim(
    types: list[dict],
    height: int,
    width: int,
    *,
    require_symmetry: bool,
    require_no_touch: bool,
    limit: int,
) -> list[np.ndarray]:
    board = np.full((height, width), -1, dtype=int)
    remaining = [int(t["count"]) for t in types]
    solutions: list[np.ndarray] = []
    seen: set[bytes] = set()

    def can_place(type_index: int, r0: int, c0: int) -> bool:
        t = types[type_index]
        color = int(t["color"])
        cells = {(r0 + dr, c0 + dc) for dr, dc in t["coords"]}

        for r, c in cells:
            if not (0 <= r < height and 0 <= c < width):
                return False
            if board[r, c] != -1:
                return False

        for r, c in cells:
            if require_symmetry:
                mirrored = int(board[r, width - 1 - c])
                if mirrored != -1 and mirrored != color:
                    return False
            if require_no_touch:
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                        if int(board[rr, cc]) == color:
                            return False
        return True

    def place(type_index: int, r0: int, c0: int) -> list[tuple[int, int]]:
        t = types[type_index]
        color = int(t["color"])
        changed = []
        for dr, dc in t["coords"]:
            r, c = r0 + dr, c0 + dc
            board[r, c] = color
            changed.append((r, c))
        remaining[type_index] -= 1
        return changed

    def unplace(type_index: int, changed: list[tuple[int, int]]) -> None:
        for r, c in changed:
            board[r, c] = -1
        remaining[type_index] += 1

    def next_empty() -> tuple[int, int] | None:
        empty = np.argwhere(board == -1)
        if empty.size == 0:
            return None
        return tuple(map(int, empty[0]))

    def placement_options(r: int, c: int) -> list[tuple[int, int, int]]:
        options = []
        for type_index, t in enumerate(types):
            if remaining[type_index] == 0:
                continue
            anchors = set()
            for dr, dc in t["coords"]:
                anchors.add((r - dr, c - dc))
            for r0, c0 in sorted(anchors):
                if can_place(type_index, r0, c0):
                    options.append((type_index, r0, c0))
        options.sort(key=lambda item: (-len(types[item[0]]["coords"]), item[1], item[2], item[0]))
        return options

    def backtrack() -> None:
        if len(solutions) >= limit:
            return
        empty = next_empty()
        if empty is None:
            if not require_symmetry or np.array_equal(board, board[:, ::-1]):
                key = board.tobytes()
                if key not in seen:
                    seen.add(key)
                    solutions.append(board.copy())
            return

        r, c = empty
        for type_index, r0, c0 in placement_options(r, c):
            changed = place(type_index, r0, c0)
            backtrack()
            unplace(type_index, changed)
            if len(solutions) >= limit:
                return

    backtrack()
    return solutions


def _count_assemblies(
    pieces: list[dict],
    *,
    dims: tuple[int, int] | None,
    require_symmetry: bool,
    require_no_touch: bool,
    limit: int,
) -> list[np.ndarray]:
    total_area = sum(int(piece["area"]) for piece in pieces)
    candidate_dims = [dims] if dims is not None else _factor_pairs(total_area)
    if not candidate_dims:
        return []

    types = _pieces_to_types(pieces)
    solutions: list[np.ndarray] = []
    seen_keys: set[tuple[tuple[int, int], bytes]] = set()

    for height, width in candidate_dims:
        for board in _search_assemblies_for_dim(
            types,
            int(height),
            int(width),
            require_symmetry=require_symmetry,
            require_no_touch=require_no_touch,
            limit=limit,
        ):
            key = (tuple(map(int, board.shape)), board.tobytes())
            if key in seen_keys:
                continue
            seen_keys.add(key)
            solutions.append(board)
            if len(solutions) >= limit:
                return solutions
    return solutions


def main(grid: np.ndarray) -> np.ndarray:
    """Filter singleton noise, then assemble the remaining fragments into one left-right symmetric rectangle.

    Input structure:
    - The background is black, with scattered monochrome 4-connected colored fragments.
    - Four gray singleton corner markers sit near the canvas corners, and extra singleton colored pixels act as noise.
    - Valid fragments are the non-gray components whose area is greater than one.

    Rule:
    - Find all monochrome 4-connected components and discard gray components plus every area-1 component.
    - Sum the remaining fragment areas and consider rectangle dimensions whose product matches that area.
    - Translate the fragments without rotation or reflection so they exactly tile one rectangle that is left-right mirror-symmetric, while pixels from different fragments of the same color never 4-touch.

    Key insight:
    - Noise filtering is necessary because singleton pixels corrupt the total area; after filtering, mirror symmetry and the same-color no-touch rule collapse the jigsaw to a unique rectangle.
    """
    pieces = _extract_pieces(np.asarray(grid, dtype=int), keep_singletons=False, drop_gray=True)
    assert pieces, "expected at least one valid fragment"

    solutions = _count_assemblies(
        pieces,
        dims=None,
        require_symmetry=True,
        require_no_touch=True,
        limit=2,
    )
    assert len(solutions) == 1, "expected a unique symmetric rectangle assembly"
    return solutions[0]


def _target_output(height: int, shared_color: int, separator_color: int) -> np.ndarray:
    row = np.array(
        [
            shared_color,
            shared_color,
            separator_color,
            shared_color,
            separator_color,
            shared_color,
            shared_color,
        ],
        dtype=int,
    )
    return np.tile(row, (int(height), 1))


def _component_patches(board: np.ndarray) -> list[np.ndarray]:
    patches = []
    for comp in _extract_pieces(board, keep_singletons=True, drop_gray=True):
        coords = comp["coords"]
        r0, c0 = np.min(coords, axis=0)
        r1, c1 = np.max(coords, axis=0)
        patches.append(board[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy())
    return patches


def _scatter_scene(output: np.ndarray, noise_count: int) -> np.ndarray:
    canvas_h = random.randint(12, 16)
    canvas_w = random.randint(12, 16)
    grid = np.zeros((canvas_h, canvas_w), dtype=int)

    marker_positions = [(0, 0), (0, canvas_w - 1), (canvas_h - 1, 0), (canvas_h - 1, canvas_w - 1)]
    for r, c in marker_positions:
        grid[r, c] = GRAY

    patches = _component_patches(output)
    random.shuffle(patches)
    for patch in patches:
        r, c = random_free_location(grid, patch, transparent=BG, border=1, padding=1, padding_connectivity=8)
        paste(grid, patch, r, c, transparent=BG)

    noise_palette = [int(color) for color in PIECE_COLORS]
    for _ in range(noise_count):
        color = random.choice(noise_palette)
        noise = np.array([[color]], dtype=int)
        r, c = random_free_location(grid, noise, transparent=BG, border=1, padding=1, padding_connectivity=8)
        paste(grid, noise, r, c, transparent=BG)

    return grid


def _validate_instance(inp: np.ndarray, expected: np.ndarray) -> None:
    actual = main(inp)
    assert np.array_equal(actual, expected)

    pieces = _extract_pieces(inp, keep_singletons=False, drop_gray=True)
    full = _count_assemblies(
        pieces,
        dims=None,
        require_symmetry=True,
        require_no_touch=True,
        limit=2,
    )
    assert len(full) == 1 and np.array_equal(full[0], expected)

    no_touch = _count_assemblies(
        pieces,
        dims=tuple(map(int, expected.shape)),
        require_symmetry=True,
        require_no_touch=False,
        limit=2,
    )
    assert len(no_touch) >= 2

    no_symmetry = _count_assemblies(
        pieces,
        dims=tuple(map(int, expected.shape)),
        require_symmetry=False,
        require_no_touch=True,
        limit=2,
    )
    assert len(no_symmetry) >= 2

    noisy_pieces = _extract_pieces(inp, keep_singletons=True, drop_gray=True)
    noisy_area = sum(int(piece["area"]) for piece in noisy_pieces)
    assert not _factor_pairs(noisy_area), "noise pixels should break rectangle area inference"


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        height = random.choice(TARGET_HEIGHTS)
        noise_count = NOISE_COUNT_BY_HEIGHT[height]
        shared_color, separator_color = random.sample(PIECE_COLORS, 2)
        expected = _target_output(height, shared_color, separator_color)

        try:
            grid = _scatter_scene(expected, noise_count)
            _validate_instance(grid, expected)
        except (AssertionError, ValueError):
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
