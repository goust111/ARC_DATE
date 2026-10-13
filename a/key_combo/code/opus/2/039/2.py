from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox, crop_bbox, find_components, get_bg, paste


LABEL_COLORS = (Color.BLUE, Color.RED, Color.GREEN)
PIECE_COLORS = (Color.BLUE, Color.GREEN)
GEN_RETRY_LIMIT = 400
LABEL_RETRY_LIMIT = 200
PLACEMENT_RETRY_LIMIT = 150


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _component_patch(inner: np.ndarray, comp: dict) -> np.ndarray:
    coords = np.asarray(comp["coords"], dtype=int)
    r0, c0, r1, c1 = map(int, comp["bbox"])
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), Color.BLACK, dtype=int)
    color = int(inner[coords[0, 0], coords[0, 1]])
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = color
    return patch


def _extract_pieces(inner: np.ndarray) -> list[np.ndarray]:
    comps = find_components(
        inner,
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert len(comps) == 2, "expected exactly two 4-connected puzzle pieces"
    patches = [_component_patch(inner, comp) for comp in comps]
    patches.sort(key=lambda patch: int(patch[patch != Color.BLACK][0]))
    colors = [int(patch[patch != Color.BLACK][0]) for patch in patches]
    assert colors == [Color.BLUE, Color.GREEN], "expected one blue piece and one green piece"
    return patches


def _piece_signature(patch: np.ndarray) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    blue = tuple(map(tuple, np.argwhere(patch == Color.BLUE)))
    green = tuple(map(tuple, np.argwhere(patch == Color.GREEN)))
    return blue, green


def _assemble_densest(patches: list[np.ndarray]) -> np.ndarray:
    assert len(patches) == 2
    a, b = patches
    a_coords = np.argwhere(a != Color.BLACK)
    b_coords = np.argwhere(b != Color.BLACK)
    a_set = {tuple(map(int, rc)) for rc in a_coords}
    total_area = int(a_coords.shape[0] + b_coords.shape[0])

    best_key = None
    best_patch = None
    for dr in range(-b.shape[0], a.shape[0] + 1):
        for dc in range(-b.shape[1], a.shape[1] + 1):
            shifted_b = b_coords + np.array([dr, dc], dtype=int)
            if any((int(r), int(c)) in a_set for r, c in shifted_b):
                continue

            all_coords = np.vstack([a_coords, shifted_b])
            r0 = int(all_coords[:, 0].min())
            c0 = int(all_coords[:, 1].min())
            r1 = int(all_coords[:, 0].max())
            c1 = int(all_coords[:, 1].max())
            bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)
            holes = bbox_area - total_area

            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), Color.BLACK, dtype=int)
            a_norm = a_coords - np.array([r0, c0], dtype=int)
            b_norm = shifted_b - np.array([r0, c0], dtype=int)
            patch[a_norm[:, 0], a_norm[:, 1]] = Color.BLUE
            patch[b_norm[:, 0], b_norm[:, 1]] = Color.GREEN

            key = (holes, bbox_area, r0, c0, _piece_signature(patch))
            if best_key is None or key < best_key:
                best_key = key
                best_patch = patch

    assert best_patch is not None, "no non-overlapping packing found"
    return best_patch


def _mask_assembly(assembly: np.ndarray, row_labels: np.ndarray, col_labels: np.ndarray) -> np.ndarray:
    out = assembly.copy()
    h, w = assembly.shape
    match = np.equal.outer(row_labels[:h], col_labels[:w])
    occ = assembly != Color.BLACK
    out[occ & ~match] = Color.GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pack the two separated puzzle pieces into their densest joint shape, then mask the packed pixels by row/column label equality.

    Input structure:
    - A black grid uses its last column as row labels and its last row as column labels.
    - The inner area contains exactly two 4-connected monochrome puzzle pieces, one blue and one green, on black.
    - Every label is blue, red, or green, and the bottom-right corner is black.

    Rule:
    - Ignore the label row and column while extracting the two pieces.
    - Translate the pieces without rotating them and choose the non-overlapping placement whose cropped union bbox has the fewest black holes; break ties by smaller bbox area, then lexicographically smaller bbox top-left.
    - Move that packed union to the inner area's top-left corner.
    - For each packed pixel, keep its original blue/green color when its row label equals its column label; otherwise recolor it grey. Leave all labels unchanged.

    Key insight:
    - Geometry comes first and masking comes second: the densest translation-only packing decides which row and column labels each piece cell lands under, and only then does label equality decide whether that cell stays colored or turns grey.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 2
    assert int(g[-1, -1]) == Color.BLACK

    bg = int(get_bg(g))
    assert bg == Color.BLACK

    inner = g[:-1, :-1]
    row_labels = g[:-1, -1].astype(int)
    col_labels = g[-1, :-1].astype(int)
    assert set(np.unique(row_labels)).issubset(set(LABEL_COLORS))
    assert set(np.unique(col_labels)).issubset(set(LABEL_COLORS))

    assembly = _assemble_densest(_extract_pieces(inner))
    ah, aw = assembly.shape
    assert ah <= inner.shape[0] and aw <= inner.shape[1]

    out = g.copy()
    out[:-1, :-1] = Color.BLACK
    paste(out[:-1, :-1], _mask_assembly(assembly, row_labels, col_labels), 0, 0, transparent=Color.BLACK)
    return out


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    points = [0, *cuts, total]
    return [points[i + 1] - points[i] for i in range(parts)]


def _sample_label_sequence(length: int) -> np.ndarray:
    part_count = random.randint(2, min(4, length))
    runs = _random_composition(length, part_count)
    colors = [random.choice(LABEL_COLORS)]
    for _ in range(part_count - 1):
        choices = [color for color in LABEL_COLORS if color != colors[-1]]
        colors.append(random.choice(choices))
    seq = np.array([color for run, color in zip(runs, colors) for _ in range(run)], dtype=int)
    assert seq.shape == (length,)
    return seq


def _grow_compact_mask(canvas_h: int, canvas_w: int, area: int) -> np.ndarray | None:
    mask = np.zeros((canvas_h, canvas_w), dtype=bool)
    start = (random.randrange(canvas_h), random.randrange(canvas_w))
    mask[start] = True
    frontier = {
        (nr, nc)
        for nr, nc in _neighbors4(*start)
        if 0 <= nr < canvas_h and 0 <= nc < canvas_w
    }

    while int(mask.sum()) < area:
        weighted: list[tuple[int, int]] = []
        for r, c in frontier:
            touch = sum(
                0 <= nr < canvas_h and 0 <= nc < canvas_w and mask[nr, nc]
                for nr, nc in _neighbors4(r, c)
            )
            if touch:
                weighted.extend([(r, c)] * (touch * touch))
        if not weighted:
            return None
        r, c = random.choice(weighted)
        mask[r, c] = True
        frontier.discard((r, c))
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < canvas_h and 0 <= nc < canvas_w and not mask[nr, nc]:
                frontier.add((nr, nc))

    box = bbox(mask)
    assert box is not None
    cropped = crop_bbox(mask.astype(int), box).astype(bool)
    if min(cropped.shape) < 3:
        return None
    if int(cropped.size - cropped.sum()) > max(4, area // 3):
        return None
    return cropped


def _partition_mask(mask: np.ndarray) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    cell_set = set(cells)
    root = random.choice(cells)

    children = {cell: [] for cell in cells}
    parent = {root: None}
    order = [root]
    stack = [root]
    while stack:
        cell = stack.pop()
        neighbors = [nbr for nbr in _neighbors4(*cell) if nbr in cell_set]
        random.shuffle(neighbors)
        for nbr in neighbors:
            if nbr in parent:
                continue
            parent[nbr] = cell
            children[cell].append(nbr)
            order.append(nbr)
            stack.append(nbr)

    if len(order) != len(cells):
        return None

    subtrees: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for cell in reversed(order):
        nodes = {cell}
        for child in children[cell]:
            nodes.update(subtrees[child])
        subtrees[cell] = nodes

    total = len(cells)
    candidates = []
    for cell in order[1:]:
        part_a = subtrees[cell]
        size_a = len(part_a)
        size_b = total - size_a
        if 6 <= size_a <= 12 and 6 <= size_b <= 12:
            candidates.append(part_a)
    if not candidates:
        return None

    a = random.choice(candidates)
    b = cell_set - a
    return a, b


def _cells_to_patch(cells: set[tuple[int, int]], color: int) -> np.ndarray:
    coords = np.array(sorted(cells), dtype=int)
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), Color.BLACK, dtype=int)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = int(color)
    return patch


def _make_piece_pair() -> list[np.ndarray] | None:
    canvas_h = random.randint(4, 6)
    canvas_w = random.randint(4, 6)
    max_area = min(canvas_h * canvas_w, 20)
    min_area = max(12, (2 * canvas_h * canvas_w) // 3)
    if min_area > max_area:
        return None
    union = _grow_compact_mask(canvas_h, canvas_w, random.randint(min_area, max_area))
    if union is None:
        return None
    parts = _partition_mask(union)
    if parts is None:
        return None
    first, second = parts
    if random.random() < 0.5:
        first, second = second, first
    patches = [_cells_to_patch(first, Color.BLUE), _cells_to_patch(second, Color.GREEN)]
    if any(not (6 <= int(np.count_nonzero(patch)) <= 12) for patch in patches):
        return None
    if any(min(patch.shape) < 2 for patch in patches):
        return None
    return patches


def _translated_coords(local_coords: np.ndarray, r0: int, c0: int) -> set[tuple[int, int]]:
    return {(int(r0 + r), int(c0 + c)) for r, c in local_coords}


def _separated(coords_a: set[tuple[int, int]], coords_b: set[tuple[int, int]]) -> bool:
    if coords_a & coords_b:
        return False
    for r, c in coords_b:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (r + dr, c + dc) in coords_a:
                    return False
    return True


def _place_pieces(inner_shape: tuple[int, int], patches: list[np.ndarray]) -> np.ndarray | None:
    H, W = map(int, inner_shape)
    coords = [np.argwhere(patch != Color.BLACK) for patch in patches]
    positions = []
    for patch in patches:
        h, w = patch.shape
        if h > H or w > W:
            return None
        pos = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
        random.shuffle(pos)
        positions.append(pos)

    for r1, c1 in positions[0][:PLACEMENT_RETRY_LIMIT]:
        abs_a = _translated_coords(coords[0], r1, c1)
        for r2, c2 in positions[1][:PLACEMENT_RETRY_LIMIT]:
            abs_b = _translated_coords(coords[1], r2, c2)
            if not _separated(abs_a, abs_b):
                continue
            inner = np.full((H, W), Color.BLACK, dtype=int)
            paste(inner, patches[0], r1, c1, transparent=Color.BLACK)
            paste(inner, patches[1], r2, c2, transparent=Color.BLACK)
            return inner
    return None


def _sample_labels(inner_shape: tuple[int, int], assembly: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    H, W = map(int, inner_shape)
    ah, aw = assembly.shape
    area = int(np.count_nonzero(assembly))
    occupied = assembly != Color.BLACK
    force_red_match = random.random() < 0.6

    for _ in range(LABEL_RETRY_LIMIT):
        row_labels = _sample_label_sequence(H)
        col_labels = _sample_label_sequence(W)
        if len(set(row_labels.tolist())) < 2 or len(set(col_labels.tolist())) < 2:
            continue
        if len(set(row_labels.tolist()) | set(col_labels.tolist())) < 3:
            continue

        match = np.equal.outer(row_labels[:ah], col_labels[:aw])
        kept = occupied & match
        grey = occupied & ~match
        if int(kept.sum()) < max(2, area // 5):
            continue
        if int(grey.sum()) < max(2, area // 5):
            continue
        if force_red_match:
            red_match = match & (row_labels[:ah, None] == Color.RED)
            if not bool((occupied & red_match).any()):
                continue
        good = True
        for color in PIECE_COLORS:
            cells = assembly == color
            if not bool((cells & kept).any()) or not bool((cells & grey).any()):
                good = False
                break
        if good:
            return row_labels, col_labels
    return None


def _render_expected(inner_shape: tuple[int, int], assembly: np.ndarray, row_labels: np.ndarray, col_labels: np.ndarray) -> np.ndarray:
    H, W = map(int, inner_shape)
    out = np.full((H + 1, W + 1), Color.BLACK, dtype=int)
    out[:-1, -1] = row_labels
    out[-1, :-1] = col_labels
    paste(out[:-1, :-1], _mask_assembly(assembly, row_labels, col_labels), 0, 0, transparent=Color.BLACK)
    return out


def _gen_attempt() -> np.ndarray | None:
    inner_shape = (random.randint(10, 13), random.randint(10, 13))
    patches = _make_piece_pair()
    if patches is None:
        return None

    assembly = _assemble_densest(patches)
    if assembly.shape[0] > inner_shape[0] or assembly.shape[1] > inner_shape[1]:
        return None

    inner = _place_pieces(inner_shape, patches)
    if inner is None:
        return None

    labels = _sample_labels(inner_shape, assembly)
    if labels is None:
        return None
    row_labels, col_labels = labels

    grid = np.full((inner_shape[0] + 1, inner_shape[1] + 1), Color.BLACK, dtype=int)
    grid[:-1, :-1] = inner
    grid[:-1, -1] = row_labels
    grid[-1, :-1] = col_labels

    comps = find_components(
        inner,
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    if len(comps) != 2:
        return None

    expected = _render_expected(inner_shape, assembly, row_labels, col_labels)
    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
