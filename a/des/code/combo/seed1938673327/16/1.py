from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in _N4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            components.append(
                {
                    "coords": np.array(coords, dtype=int),
                    "bbox": _bbox(coords),
                    "area": len(coords),
                }
            )

    return components


def _extract_legend(grid: np.ndarray, components: list[dict[str, object]], bg: int) -> tuple[dict[str, object], list[int]]:
    h, w = grid.shape
    candidates: list[dict[str, object]] = []

    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]  # type: ignore[index]
        area = int(comp["area"])  # type: ignore[index]
        bh = int(r1 - r0 + 1)
        bw = int(c1 - c0 + 1)
        touches_edge = r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1
        if not touches_edge:
            continue
        if area != 4:
            continue
        if not ((bh == 1 and bw == 4) or (bh == 4 and bw == 1)):
            continue

        coords = [tuple(map(int, rc)) for rc in comp["coords"]]  # type: ignore[index]
        colors = {int(grid[r, c]) for r, c in coords}
        if bg in colors or len(colors) != 4:
            continue
        candidates.append(comp)

    assert len(candidates) == 1, "expected exactly one edge legend strip"
    legend = candidates[0]
    r0, c0, r1, c1 = legend["bbox"]  # type: ignore[index]
    coords = [tuple(map(int, rc)) for rc in legend["coords"]]  # type: ignore[index]
    if r0 == r1:
        coords.sort(key=lambda rc: rc[1])
    else:
        coords.sort(key=lambda rc: rc[0])
    colors = [int(grid[r, c]) for r, c in coords]
    return legend, colors


def _rotated_anchor(bbox: tuple[int, int, int, int], rotated_shape: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    new_h, new_w = rotated_shape

    center_r = r0 + (h - 1) / 2.0
    center_c = c0 + (w - 1) / 2.0
    new_r0 = center_r - (new_h - 1) / 2.0
    new_c0 = center_c - (new_w - 1) / 2.0

    assert abs(new_r0 - round(new_r0)) < 1e-9
    assert abs(new_c0 - round(new_c0)) < 1e-9
    return int(round(new_r0)), int(round(new_c0))


def _paste_patch(out: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    assert 0 <= r0 and 0 <= c0 and r0 + h <= out.shape[0] and c0 + w <= out.shape[1]

    mask = patch != bg
    target = out[r0 : r0 + h, c0 : c0 + w]
    assert not np.any((target != bg) & mask), "rotated objects must remain disjoint"
    target[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each foreground object according to the color of its single special pixel.

    Input structure
    - The grid has a dominant black background.
    - A four-cell color legend lies on one grid edge and encodes 0/90/180/270 degrees from left to right.
    - Every other 4-connected foreground object mostly uses one main color and contains exactly one special pixel of a legend color.

    Rule
    - For each object, find the unique special-pixel color.
    - Read that color's position in the legend to get the rotation angle.
    - Rotate the full tight object patch around its bbox center and redraw it.
    - Keep the legend unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    components = _find_components(g, bg)
    assert components, "expected at least one foreground component"

    legend, legend_colors = _extract_legend(g, components, bg)
    legend_id = id(legend)

    out = np.full_like(g, bg)
    for r, c in legend["coords"]:  # type: ignore[index]
        rr, cc = map(int, (r, c))
        out[rr, cc] = g[rr, cc]

    for comp in components:
        if id(comp) == legend_id:
            continue

        coords = [tuple(map(int, rc)) for rc in comp["coords"]]  # type: ignore[index]
        color_counts = Counter(int(g[r, c]) for r, c in coords)
        assert len(color_counts) == 2, "each object should use one main color and one special color"
        special = [color for color, count in color_counts.items() if count == 1]
        assert len(special) == 1, "expected exactly one special pixel"
        special_color = int(special[0])
        assert special_color in legend_colors, "special color must appear in legend"

        turns = legend_colors.index(special_color)
        r0, c0, r1, c1 = comp["bbox"]  # type: ignore[index]
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        rotated = np.rot90(patch, turns)
        nr0, nc0 = _rotated_anchor((int(r0), int(c0), int(r1), int(c1)), rotated.shape)
        _paste_patch(out, rotated, nr0, nc0, bg)

    return out


# --------------- generator ---------------

_GEN_BG = 0
_GEN_PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
_GEN_GRID_H = (12, 18)
_GEN_GRID_W = (12, 18)
_GEN_OBJECT_COUNT = (2, 4)
_GEN_SIZES = [3, 5, 7]
_GEN_BOX_GAP = 1
_GEN_RETRIES = 400


def _path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    er, ec = end
    moves = []
    moves.extend([(1 if er > r else -1, 0)] * abs(er - r))
    moves.extend([(0, 1 if ec > c else -1)] * abs(ec - c))
    random.shuffle(moves)

    cells = [(r, c)]
    for dr, dc in moves:
        r += dr
        c += dc
        cells.append((r, c))
    return cells


def _random_mask(size: int) -> np.ndarray | None:
    if size < 3:
        return None

    center = (random.randint(1, size - 2), random.randint(1, size - 2))
    anchors = [
        (0, random.randint(0, size - 1)),
        (size - 1, random.randint(0, size - 1)),
        (random.randint(0, size - 1), 0),
        (random.randint(0, size - 1), size - 1),
    ]

    cells = {center}
    for anchor in anchors:
        cells.update(_path(center, anchor))

    max_area = size * size - 2
    if len(cells) > max_area:
        return None

    target_area = random.randint(len(cells), max_area)
    while len(cells) < target_area:
        frontier = set()
        for r, c in cells:
            for dr, dc in _N4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(tuple(frontier)))

    mask = np.zeros((size, size), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    if not mask[0, :].any() or not mask[-1, :].any() or not mask[:, 0].any() or not mask[:, -1].any():
        return None

    return mask


def _special_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    candidates = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if mask[r, c]:
                candidates.append((r, c))
    return candidates


def _make_object_patch(size: int, main_color: int, special_color: int, turns: int) -> np.ndarray | None:
    for _ in range(100):
        mask = _random_mask(size)
        if mask is None:
            continue

        special_cells = _special_candidates(mask)
        if not special_cells:
            continue

        center = (size // 2, size // 2)
        if turns != 0:
            off_center = [cell for cell in special_cells if cell != center]
            if off_center:
                special_cells = off_center
        sr, sc = random.choice(special_cells)

        patch = np.full((size, size), _GEN_BG, dtype=int)
        patch[mask] = main_color
        patch[sr, sc] = special_color

        counts = Counter(int(v) for v in patch[patch != _GEN_BG])
        if sorted(counts.values()) != [1, int(mask.sum()) - 1]:
            continue
        if turns != 0 and np.array_equal(np.rot90(patch, turns), patch):
            continue
        return patch

    return None


def _separated(a: tuple[int, int, int], b: tuple[int, int, int], gap: int) -> bool:
    ar, ac, asz = a
    br, bc, bsz = b
    ar1, ac1 = ar + asz - 1, ac + asz - 1
    br1, bc1 = br + bsz - 1, bc + bsz - 1
    return ar1 + gap < br or br1 + gap < ar or ac1 + gap < bc or bc1 + gap < ac


def _place_boxes(height: int, width: int, legend_side: str, sizes: list[int]) -> list[tuple[int, int, int]] | None:
    top_block = 2 if legend_side == "top" else 0
    bottom_block = 2 if legend_side == "bottom" else 0
    positions: list[tuple[int, int, int]] = []

    for size in sorted(sizes, reverse=True):
        candidates = []
        for r in range(top_block, height - bottom_block - size + 1):
            for c in range(0, width - size + 1):
                cand = (r, c, size)
                if all(_separated(cand, other, _GEN_BOX_GAP) for other in positions):
                    candidates.append(cand)
        if not candidates:
            return None
        positions.append(random.choice(candidates))

    random.shuffle(positions)
    return positions


def _expected_output(grid: np.ndarray, legend_cells: list[tuple[int, int]], objects: list[dict[str, object]]) -> np.ndarray:
    out = np.full_like(grid, _GEN_BG)
    for r, c in legend_cells:
        out[r, c] = grid[r, c]
    for obj in objects:
        r0, c0 = obj["pos"]  # type: ignore[misc]
        patch = obj["patch"]  # type: ignore[assignment]
        turns = int(obj["turns"])  # type: ignore[arg-type]
        rotated = np.rot90(patch, turns)
        out[r0 : r0 + rotated.shape[0], c0 : c0 + rotated.shape[1]][rotated != _GEN_BG] = rotated[rotated != _GEN_BG]
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*_GEN_GRID_H)
    width = random.randint(*_GEN_GRID_W)
    object_count = random.randint(*_GEN_OBJECT_COUNT)
    legend_side = random.choice(["top", "bottom"])

    if width < 6:
        return None

    legend_colors = random.sample(_GEN_PALETTE, 4)
    object_turns = random.sample([0, 1, 2, 3], object_count)
    main_colors = random.sample([c for c in _GEN_PALETTE if c not in legend_colors], object_count)
    sizes = random.choices(_GEN_SIZES, k=object_count)

    placements = _place_boxes(height, width, legend_side, sizes)
    if placements is None:
        return None

    grid = np.full((height, width), _GEN_BG, dtype=int)
    legend_row = 0 if legend_side == "top" else height - 1
    legend_c0 = random.randint(0, width - 4)
    legend_cells = [(legend_row, legend_c0 + i) for i in range(4)]
    for i, (r, c) in enumerate(legend_cells):
        grid[r, c] = legend_colors[i]

    objects: list[dict[str, object]] = []
    for (r0, c0, size), main_color, turns in zip(placements, main_colors, object_turns):
        patch = _make_object_patch(size, main_color, legend_colors[turns], turns)
        if patch is None:
            return None
        if np.any(grid[r0 : r0 + size, c0 : c0 + size] != _GEN_BG):
            return None
        grid[r0 : r0 + size, c0 : c0 + size][patch != _GEN_BG] = patch[patch != _GEN_BG]
        objects.append({"pos": (r0, c0), "patch": patch, "turns": turns})

    expected = _expected_output(grid, legend_cells, objects)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
