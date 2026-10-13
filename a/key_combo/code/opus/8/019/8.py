from __future__ import annotations

import random

import numpy as np


BG = 0
FG = 1
MARKER = 8
MAX_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _marker_frame(grid: np.ndarray) -> tuple[int, int, int, int, int]:
    coords = np.argwhere(grid == MARKER)
    assert len(coords) == 4, "expected exactly four corner markers"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    corners = {(int(r0), int(c0)), (int(r0), int(c1)), (int(r1), int(c0)), (int(r1), int(c1))}
    assert {tuple(map(int, p)) for p in coords.tolist()} == corners, "markers must be rectangle corners"
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "marker rectangle must enclose a non-empty interior"
    return int(r0), int(c0), int(r1), int(c1), MARKER


def _fourfold_from_source(source: np.ndarray) -> np.ndarray:
    top = np.concatenate([source, np.fliplr(source)], axis=1)
    bottom = np.concatenate([np.flipud(source), np.flipud(np.fliplr(source))], axis=1)
    return np.concatenate([top, bottom], axis=0)


def _hole_regions(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []

    for r in range(h):
        for c in (0, w - 1):
            if grid[r, c] == BG and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if grid[r, c] == BG and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == BG and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))

    holes: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != BG or seen[r, c]:
                continue
            comp = []
            seen[r, c] = True
            stack = [(r, c)]
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == BG and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            holes.append(comp)
    return holes


def _fill_holes(grid: np.ndarray, color: int) -> np.ndarray:
    out = grid.copy()
    for comp in _hole_regions(out):
        for r, c in comp:
            out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the marker-bounded quadrant seed, complete it by fourfold symmetry, then fill the emergent enclosed holes with the marker color.

    Input structure:
    - Black background with four teal corner markers defining a rectangle, but no drawn edges between them.
    - The rectangle interior has even height and width.
    - Only the interior's upper-left quadrant contains a blue foreground pattern; the other three quadrants are black.

    Rule:
    - Use the four teal markers to locate the rectangle and crop only its interior, excluding the marker rows and columns.
    - Take the cropped upper-left quadrant as the source pattern and mirror it left-right, up-down, and both to reconstruct all four quadrants.
    - Flood-fill from the reconstructed crop boundary through black cells using 4-connectivity; any remaining black regions are enclosed holes.
    - Recolor every enclosed hole with the remembered teal marker color and return the cropped completed pattern.

    Key insight:
    - The seed quadrant is open on its right and bottom sides, so it has no hole by itself; only the fourfold symmetric completion closes those openings into a central hole that must be filled with the marker color that disappeared during cropping.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    r0, c0, r1, c1, fill_color = _marker_frame(g)
    cropped = g[r0 + 1 : r1, c0 + 1 : c1].copy()
    h, w = cropped.shape
    assert h > 0 and w > 0 and h % 2 == 0 and w % 2 == 0, "cropped interior must be even-sized"

    source = cropped[: h // 2, : w // 2]
    completed = _fourfold_from_source(source)
    return _fill_holes(completed, fill_color)


def _make_source(qh: int, qw: int) -> np.ndarray | None:
    options = []
    for r0 in range(qh - 1):
        for c0 in range(qw - 1):
            base = (qh - r0) + (qw - c0) - 1
            if 3 <= base <= 8:
                options.append((r0, c0, base))
    if not options:
        return None

    r0, c0, base = random.choice(options)
    source = np.zeros((qh, qw), dtype=int)
    source[r0, c0:] = FG
    source[r0:, c0] = FG

    budget = 8 - base
    branch_modes = []
    if r0 > 0:
        branch_modes.append("up")
    if c0 > 0:
        branch_modes.append("left")
    random.shuffle(branch_modes)

    if branch_modes and budget > 0:
        target_branch_count = random.randint(0, min(2, len(branch_modes)))
        used = 0
        for mode in branch_modes[:target_branch_count]:
            remaining = budget - used
            if remaining <= 0:
                break
            if mode == "up":
                col = random.randint(c0, qw - 1)
                length = random.randint(1, min(r0, remaining))
                source[r0 - length : r0, col] = FG
                used += length
            else:
                row = random.randint(r0, qh - 1)
                length = random.randint(1, min(c0, remaining))
                source[row, c0 - length : c0] = FG
                used += length

    if int(np.count_nonzero(source == FG)) < 3 or int(np.count_nonzero(source == FG)) > 8:
        return None
    if np.array_equal(source, np.fliplr(source)) or np.array_equal(source, np.flipud(source)):
        return None
    if not np.any(source[:, -1] == FG) or not np.any(source[-1, :] == FG):
        return None
    if _hole_regions(source):
        return None
    return source


def _generate_crop() -> tuple[np.ndarray, np.ndarray] | None:
    qh = random.randint(3, 6)
    qw = random.randint(3, 6)
    source = _make_source(qh, qw)
    if source is None:
        return None

    crop = np.zeros((2 * qh, 2 * qw), dtype=int)
    crop[:qh, :qw] = source

    completed = _fourfold_from_source(source)
    holes = _hole_regions(completed)
    if not holes or max(len(comp) for comp in holes) < 2:
        return None

    expected = _fill_holes(completed, MARKER)
    if not np.any(expected == MARKER):
        return None
    return crop, expected


def _embed_with_markers(crop: np.ndarray) -> np.ndarray:
    crop_h, crop_w = crop.shape
    top = random.randint(1, 5)
    left = random.randint(1, 5)
    bottom = random.randint(top, min(6, 29 - crop_h - 2 - top))
    right = random.randint(left, min(6, 29 - crop_w - 2 - left))

    h = crop_h + top + bottom + 2
    w = crop_w + left + right + 2
    grid = np.zeros((h, w), dtype=int)

    r0 = top
    c0 = left
    r1 = r0 + crop_h + 1
    c1 = c0 + crop_w + 1
    grid[r0, c0] = MARKER
    grid[r0, c1] = MARKER
    grid[r1, c0] = MARKER
    grid[r1, c1] = MARKER
    grid[r0 + 1 : r1, c0 + 1 : c1] = crop
    return grid


def _semantic_checks(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        r0, c0, r1, c1, _fill = _marker_frame(inp)
    except AssertionError:
        return False

    crop = inp[r0 + 1 : r1, c0 + 1 : c1]
    h, w = crop.shape
    if h % 2 != 0 or w % 2 != 0:
        return False
    qh, qw = h // 2, w // 2
    if np.any(crop[:qh, qw:] != BG) or np.any(crop[qh:, :] != BG):
        return False

    source = crop[:qh, :qw]
    if _hole_regions(source):
        return False
    if np.array_equal(source, np.fliplr(source)) or np.array_equal(source, np.flipud(source)):
        return False

    rebuilt = _fourfold_from_source(source)
    holes = _hole_regions(rebuilt)
    if not holes or max(len(comp) for comp in holes) < 2:
        return False
    if not np.array_equal(_fill_holes(rebuilt, MARKER), expected):
        return False
    if not np.array_equal(main(inp), expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    generated = _generate_crop()
    if generated is None:
        return None
    crop, expected = generated
    inp = _embed_with_markers(crop)
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    if not _semantic_checks(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
