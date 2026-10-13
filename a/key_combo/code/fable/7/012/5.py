"""ARC task: rotate and dock complementary pieces into toothed frames."""

from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as (row, column) coordinate arrays."""
    seen = np.zeros(mask.shape, dtype=bool)
    result: list[np.ndarray] = []
    height, width = mask.shape
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        queue = [(sr, sc)]
        cells = []
        while queue:
            r, c = queue.pop()
            cells.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _is_frame_component(grid: np.ndarray, color: int, coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 4 or c1 - c0 < 4:
        return False
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    perimeter = np.concatenate((patch[0], patch[-1], patch[1:-1, 0], patch[1:-1, -1]))
    interior = patch[1:-1, 1:-1]
    return bool(np.all(perimeter == color) and np.any(interior == color) and np.any(interior != color))


def _unique_rotations(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        if not any(rotated.shape == old.shape and np.array_equal(rotated, old) for old in variants):
            variants.append(rotated)
    return variants


def _parse(grid: np.ndarray):
    values = Counter(map(int, grid.ravel()))
    bg = values.most_common(1)[0][0]
    colors = [int(v) for v in np.unique(grid) if int(v) != bg]

    by_color: dict[int, list[np.ndarray]] = {
        color: _components(grid == color) for color in colors
    }
    frame_groups = []
    for color, comps in by_color.items():
        candidates = [comp for comp in comps if _is_frame_component(grid, color, comp)]
        if len(candidates) >= 2 and len(candidates) == len(comps):
            frame_groups.append((color, candidates))
    assert len(frame_groups) == 1, "expected one repeated hollow-frame color"
    frame_color, frame_components = frame_groups[0]

    frames = []
    for coords in frame_components:
        r0, c0, r1, c1 = _bbox(coords)
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        teeth = interior == frame_color
        assert teeth.any() and (~teeth).any()
        frames.append({"bbox": (r0, c0, r1, c1), "teeth": teeth})

    pieces = []
    for color, comps in by_color.items():
        if color == frame_color:
            continue
        for coords in comps:
            r0, c0, r1, c1 = _bbox(coords)
            mask = grid[r0 : r1 + 1, c0 : c1 + 1] == color
            pieces.append({"color": color, "coords": coords, "mask": mask})
    assert len(pieces) == len(frames) >= 2, "expected one free piece per frame"
    return bg, frame_color, frames, pieces


def _matching(frames, pieces):
    candidates: list[list[tuple[int, np.ndarray]]] = []
    for piece in pieces:
        options = []
        for rotated in _unique_rotations(piece["mask"]):
            for fi, frame in enumerate(frames):
                complement = ~frame["teeth"]
                if rotated.shape == complement.shape and np.array_equal(rotated, complement):
                    options.append((fi, rotated))
        candidates.append(options)
    assert all(len(options) == 1 for options in candidates), "each piece needs one frame and orientation"
    frame_ids = [options[0][0] for options in candidates]
    assert sorted(frame_ids) == list(range(len(frames))), "matching must be one-to-one"
    return [options[0] for options in candidates]


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate and translate every free object into the uniquely complementary toothed frame.

    Input structure:
    - The dominant background contains at least two disconnected one-cell-thick rectangular frames of one shared color.
    - Frame-colored teeth protrude from each wall into its rectangular interior.
    - An equal number of disconnected monochromatic free objects have distinct non-frame colors.

    Rule:
    - For every free object, test its four right-angle rotations against every frame interior.
    - Select the unique rotation and frame for which object occupancy is exactly the complement of the teeth.
    - Erase the object's source cells and paint that rotated object into the interior; preserve frame walls and teeth.

    Key insight:
    - Mere collision-free containment is insufficient: the chosen object and the lock teeth must jointly occupy every interior cell exactly once.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg, _frame_color, frames, pieces = _parse(g)
    matches = _matching(frames, pieces)

    out = g.copy()
    for piece in pieces:
        coords = piece["coords"]
        out[coords[:, 0], coords[:, 1]] = bg
    for piece, (frame_index, rotated) in zip(pieces, matches):
        r0, c0, r1, c1 = frames[frame_index]["bbox"]
        target = out[r0 + 1 : r1, c0 + 1 : c1]
        assert target.shape == rotated.shape
        assert np.all(target[rotated] == bg)
        target[rotated] = piece["color"]
    return out


def _connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1


def _sample_tooth_chain(height: int, width: int, n: int) -> list[np.ndarray] | None:
    arms: list[set[tuple[int, int]]] = []
    choices = []
    for side in range(4):
        span = width if side < 2 else height
        for pos in range(span):
            for depth in (1, 2):
                if side == 0:
                    cells = {(d, pos) for d in range(min(depth, height))}
                elif side == 1:
                    cells = {(height - 1 - d, pos) for d in range(min(depth, height))}
                elif side == 2:
                    cells = {(pos, d) for d in range(min(depth, width))}
                else:
                    cells = {(pos, width - 1 - d) for d in range(min(depth, width))}
                choices.append(cells)
    random.shuffle(choices)
    used: set[tuple[int, int]] = set()
    # Each successive lock has an extra visible wall-attached tooth.
    while choices and len(arms) < n + 2:
        arm = choices.pop()
        new = arm - used
        if not new:
            continue
        trial = used | arm
        complement = np.ones((height, width), dtype=bool)
        for r, c in trial:
            complement[r, c] = False
        if not _connected(complement):
            continue
        if not (complement[0].any() and complement[-1].any() and complement[:, 0].any() and complement[:, -1].any()):
            continue
        arms.append(arm)
        used = trial
    if len(arms) < n + 1:
        return None

    chain = []
    used = set()
    # Begin with two arms so every frame has plural, visually salient teeth.
    for arm in arms[:2]:
        used |= arm
    for i in range(n):
        if i:
            used |= arms[i + 1]
        teeth = np.zeros((height, width), dtype=bool)
        for r, c in used:
            teeth[r, c] = True
        chain.append(teeth)

    complements = [~teeth for teeth in chain]
    if any(not _connected(mask) for mask in complements):
        return None
    # No two locks may become equivalent after rotation, and each orientation is visible.
    if any(len(_unique_rotations(mask)) < 4 for mask in complements):
        return None
    for i, mask in enumerate(complements):
        for j, other in enumerate(complements):
            if i != j and any(rot.shape == other.shape and np.array_equal(rot, other) for rot in _unique_rotations(mask)):
                return None
    return chain


def _frame_patch(teeth: np.ndarray, color: int, bg: int) -> np.ndarray:
    h, w = teeth.shape
    patch = np.full((h + 2, w + 2), bg, dtype=int)
    patch[0, :] = patch[-1, :] = color
    patch[:, 0] = patch[:, -1] = color
    patch[1:-1, 1:-1][teeth] = color
    return patch


def _locations(grid: np.ndarray, patch: np.ndarray, bg: int, padding: int = 1):
    h, w = patch.shape
    occupied = grid != bg
    result = []
    for r in range(1, grid.shape[0] - h):
        for c in range(1, grid.shape[1] - w):
            r0, r1 = max(0, r - padding), min(grid.shape[0], r + h + padding)
            c0, c1 = max(0, c - padding), min(grid.shape[1], c + w + padding)
            if not occupied[r0:r1, c0:c1].any():
                result.append((r, c))
    return result


def _semantic_check(inp: np.ndarray, expected: np.ndarray) -> bool:
    """Check the described rule, uniqueness, and the intended anti-shortcut case."""
    try:
        bg, _fc, frames, pieces = _parse(inp)
        matches = _matching(frames, pieces)
        if not np.array_equal(main(inp), expected):
            return False
        if np.array_equal(inp, expected):
            return False
        # At least one source genuinely needs rotation, not translation alone.
        if not any(not np.array_equal(piece["mask"], rotated) for piece, (_fi, rotated) in zip(pieces, matches)):
            return False
        # At least one wrong frame admits collision-free containment but leaves gaps.
        loose_fit = False
        for pi, piece in enumerate(pieces):
            true_frame = matches[pi][0]
            for rotated in _unique_rotations(piece["mask"]):
                for fi, frame in enumerate(frames):
                    if fi == true_frame or rotated.shape != frame["teeth"].shape:
                        continue
                    if not np.any(rotated & frame["teeth"]) and not np.array_equal(rotated, ~frame["teeth"]):
                        loose_fit = True
        if not loose_fit:
            return False
        # Erasure is necessary: every original piece cell is background afterward.
        if any(np.any(expected[p["coords"][:, 0], p["coords"][:, 1]] != bg) for p in pieces):
            return False
        return True
    except (AssertionError, ValueError):
        return False


def _generate_attempt() -> np.ndarray | None:
    n = random.choice((2, 3))
    inner_h = random.randint(4, 6)
    inner_w = random.randint(4, 6)
    chain = _sample_tooth_chain(inner_h, inner_w, n)
    if chain is None:
        return None

    frame_h, frame_w = inner_h + 2, inner_w + 2
    min_width = n * frame_w + (n - 1) * 2 + 2
    if min_width > 30:
        return None
    width = random.randint(min_width, 30)
    min_height = 2 * frame_h + 5
    if min_height > 30:
        return None
    height = random.randint(min_height, min(30, min_height + 5))
    bg = 0
    palette = random.sample(range(1, 10), n + 1)
    frame_color, piece_colors = palette[0], palette[1:]
    grid = np.full((height, width), bg, dtype=int)
    expected = grid.copy()

    # Place locks in a clearly separated upper row, with randomized horizontal slack.
    slack = width - (n * frame_w + (n - 1) * 2)
    left = random.randint(1, max(1, slack - 1))
    frame_row = random.randint(1, 2)
    frame_boxes = []
    c = left
    for teeth in chain:
        patch = _frame_patch(teeth, frame_color, bg)
        grid[frame_row : frame_row + frame_h, c : c + frame_w] = patch
        expected[frame_row : frame_row + frame_h, c : c + frame_w] = patch
        frame_boxes.append((frame_row, c, frame_row + frame_h - 1, c + frame_w - 1))
        c += frame_w + 2

    rotations = [random.randrange(4) for _ in range(n)]
    if all(k == 0 for k in rotations):
        rotations[random.randrange(n)] = random.choice((1, 2, 3))
    order = list(range(n))
    random.shuffle(order)
    for i in order:
        source_mask = np.rot90(~chain[i], rotations[i])
        patch = np.full(source_mask.shape, bg, dtype=int)
        patch[source_mask] = piece_colors[i]
        spots = [(r, c) for r, c in _locations(grid, patch, bg, padding=1) if r > frame_row + frame_h + 1]
        if not spots:
            return None
        r, c = random.choice(spots)
        view = grid[r : r + patch.shape[0], c : c + patch.shape[1]]
        view[source_mask] = piece_colors[i]

        fr0, fc0, fr1, fc1 = frame_boxes[i]
        target = expected[fr0 + 1 : fr1, fc0 + 1 : fc1]
        target[~chain[i]] = piece_colors[i]

    # A whole-grid turn makes layouts and all directional teeth equally represented.
    k_canvas = random.randrange(4)
    grid = np.rot90(grid, k_canvas).copy()
    expected = np.rot90(expected, k_canvas).copy()
    return grid if _semantic_check(grid, expected) else None


def generate_input() -> np.ndarray:
    """Generate varied, uniquely solvable two- or three-lock instances."""
    for _ in range(600):
        candidate = _generate_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid lock-complement instance")
