import math
import random
from collections import Counter

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 300


UNIQUE_TEMPLATES = [
    [
        np.array([[1, 1, 0], [1, 1, 0], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 1], [0, 1, 1]], dtype=bool),
        np.array([[1, 1, 1], [1, 1, 0], [1, 1, 0]], dtype=bool),
        np.array([[0, 0, 1], [1, 1, 1], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 0], [1, 1, 1], [1, 1, 1]], dtype=bool),
    ],
    [
        np.array([[1, 1, 1], [1, 1, 1], [0, 0, 1]], dtype=bool),
        np.array([[1, 1, 0], [1, 1, 0], [1, 1, 1]], dtype=bool),
        np.array([[0, 1, 1], [0, 1, 1], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 1], [1, 1, 0], [1, 1, 0]], dtype=bool),
        np.array([[1, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=bool),
    ],
    [
        np.array([[1, 0, 0], [1, 1, 1], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 1], [1, 1, 1], [1, 0, 0]], dtype=bool),
        np.array([[0, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 1], [0, 1, 1]], dtype=bool),
        np.array([[0, 0, 1], [1, 1, 1], [1, 1, 1]], dtype=bool),
    ],
    [
        np.array([[1, 1, 0], [1, 1, 1], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 1], [1, 1, 0], [1, 1, 0]], dtype=bool),
        np.array([[1, 1, 1], [1, 1, 1], [0, 0, 1]], dtype=bool),
        np.array([[1, 0, 0], [1, 1, 1], [1, 1, 1]], dtype=bool),
        np.array([[0, 0, 1], [1, 1, 1], [1, 1, 1]], dtype=bool),
    ],
]


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_coords(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere(mask)
    return tuple((int(r), int(c)) for r, c in coords)


def _extract_pieces(grid: np.ndarray) -> tuple[list[dict], int, int]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    pieces: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        for coords in _components(g == color):
            r0, c0, r1, c1 = _bbox(coords)
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            mask = patch == color
            bh, bw = patch.shape
            assert bh >= 2 and bw >= 2, "pieces must keep their full bounding box"
            holes = bh * bw - int(mask.sum())
            assert 1 <= holes <= 2, "each piece must have one or two holes"
            pieces.append(
                {
                    "color": color,
                    "bbox": (bh, bw),
                    "coords": _mask_coords(mask),
                    "area": int(mask.sum()),
                }
            )
    assert pieces, "expected at least one foreground piece"
    bbox_set = {piece["bbox"] for piece in pieces}
    assert len(bbox_set) == 1, "all pieces must share a common bounding box"
    bh, bw = next(iter(bbox_set))
    pieces.sort(key=lambda piece: piece["color"])
    return pieces, bh, bw


def _solve_fixed_dims(
    pieces: list[dict], box_h: int, box_w: int, out_h: int, out_w: int, limit: int
) -> list[np.ndarray]:
    total_area = sum(piece["area"] for piece in pieces)
    if total_area != out_h * out_w or out_h < box_h or out_w < box_w:
        return []

    n = len(pieces)
    placements_by_piece: list[list[tuple[int, ...]]] = []
    cell_to_placements: list[list[tuple[int, int]]] = [[] for _ in range(out_h * out_w)]

    for piece_idx, piece in enumerate(pieces):
        piece_placements: list[tuple[int, ...]] = []
        for r0 in range(out_h - box_h + 1):
            for c0 in range(out_w - box_w + 1):
                cells = tuple((r0 + r) * out_w + (c0 + c) for r, c in piece["coords"])
                placement_idx = len(piece_placements)
                piece_placements.append(cells)
                for cell in cells:
                    cell_to_placements[cell].append((piece_idx, placement_idx))
        placements_by_piece.append(piece_placements)

    covered = np.zeros(out_h * out_w, dtype=bool)
    painted = np.full(out_h * out_w, -1, dtype=int)
    used = [False] * n
    solutions: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()

    def rec(covered_count: int) -> None:
        if len(solutions) >= limit:
            return
        if covered_count == out_h * out_w:
            key = ((out_h, out_w), painted.tobytes())
            if key not in seen:
                seen.add(key)
                solutions.append(painted.reshape(out_h, out_w).copy())
            return

        best_options: list[tuple[int, int]] | None = None
        for cell in np.flatnonzero(~covered):
            options: list[tuple[int, int]] = []
            for piece_idx, placement_idx in cell_to_placements[int(cell)]:
                if used[piece_idx]:
                    continue
                cells = placements_by_piece[piece_idx][placement_idx]
                if any(covered[idx] for idx in cells):
                    continue
                options.append((piece_idx, placement_idx))
            if not options:
                return
            if best_options is None or len(options) < len(best_options):
                best_options = options
                if len(best_options) == 1:
                    break

        assert best_options is not None
        for piece_idx, placement_idx in best_options:
            used[piece_idx] = True
            color = pieces[piece_idx]["color"]
            changed = placements_by_piece[piece_idx][placement_idx]
            for idx in changed:
                covered[idx] = True
                painted[idx] = color
            rec(covered_count + pieces[piece_idx]["area"])
            for idx in changed:
                covered[idx] = False
                painted[idx] = -1
            used[piece_idx] = False
            if len(solutions) >= limit:
                return

    rec(0)
    return solutions


def _candidate_dims(total_area: int, box_h: int, box_w: int) -> list[tuple[int, int]]:
    dims: set[tuple[int, int]] = set()
    for h in range(1, int(math.isqrt(total_area)) + 1):
        if total_area % h != 0:
            continue
        w = total_area // h
        if h >= box_h and w >= box_w:
            dims.add((h, w))
        if w >= box_h and h >= box_w:
            dims.add((w, h))
    return sorted(dims, key=lambda rc: (abs(rc[0] - rc[1]), rc[0], rc[1]))


def _solve_outputs_from_pieces(pieces: list[dict], box_h: int, box_w: int, limit: int) -> list[np.ndarray]:
    total_area = sum(piece["area"] for piece in pieces)
    outputs: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for out_h, out_w in _candidate_dims(total_area, box_h, box_w):
        local = _solve_fixed_dims(pieces, box_h, box_w, out_h, out_w, limit=max(1, limit - len(outputs)))
        for arr in local:
            key = (arr.shape, arr.tobytes())
            if key not in seen:
                seen.add(key)
                outputs.append(arr)
                if len(outputs) >= limit:
                    return outputs
    return outputs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered same-bbox puzzle sprites into the unique solid rectangle
    that their border holes imply.
    """
    pieces, box_h, box_w = _extract_pieces(np.asarray(grid, dtype=int))
    outputs = _solve_outputs_from_pieces(pieces, box_h, box_w, limit=2)
    assert len(outputs) == 1, "expected a unique rectangular assembly"
    return outputs[0]


def _transform_mask(mask: np.ndarray, symmetry: int) -> np.ndarray:
    out = mask.copy()
    if symmetry >= 4:
        out = np.fliplr(out)
    out = np.rot90(out, symmetry % 4)
    return out.copy()


def _scatter_patches(
    patches: list[np.ndarray], bg: int, out_h: int, out_w: int
) -> np.ndarray | None:
    n = len(patches)
    box_h, box_w = patches[0].shape
    base_h = max(out_h + 4, (box_h + 2) * max(2, math.ceil(n / 3)))
    base_w = max(out_w + 6, (box_w + 2) * max(3, math.ceil(n / 2)))

    for _ in range(80):
        in_h = random.randint(base_h, base_h + 4)
        in_w = random.randint(base_w, base_w + 5)
        grid = np.full((in_h, in_w), bg, dtype=int)
        reserved = np.zeros((in_h, in_w), dtype=bool)
        order = list(range(n))
        random.shuffle(order)
        ok = True
        for idx in order:
            patch = patches[idx]
            placed = False
            for _ in range(200):
                r0 = random.randint(0, in_h - box_h)
                c0 = random.randint(0, in_w - box_w)
                rr0 = max(0, r0 - 1)
                rr1 = min(in_h, r0 + box_h + 1)
                cc0 = max(0, c0 - 1)
                cc1 = min(in_w, c0 + box_w + 1)
                if reserved[rr0:rr1, cc0:cc1].any():
                    continue
                mask = patch != bg
                grid[r0 : r0 + box_h, c0 : c0 + box_w][mask] = patch[mask]
                reserved[r0 : r0 + box_h, c0 : c0 + box_w] = True
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok and Counter(grid.ravel()).most_common(1)[0][0] == bg:
            return grid
    return None


def _build_instance() -> tuple[np.ndarray, np.ndarray] | None:
    masks = [mask.copy() for mask in random.choice(UNIQUE_TEMPLATES)]
    symmetry = random.randrange(8)
    masks = [_transform_mask(mask, symmetry) for mask in masks]
    random.shuffle(masks)

    box_h, box_w = masks[0].shape
    color_pool = list(range(10))
    chosen_colors = random.sample(color_pool, len(masks) + 1)
    bg = chosen_colors[0]
    piece_colors = chosen_colors[1:]
    random.shuffle(piece_colors)

    patches: list[np.ndarray] = []
    pieces: list[dict] = []
    for color, mask in zip(piece_colors, masks):
        patch = np.full((box_h, box_w), bg, dtype=int)
        patch[mask] = color
        patches.append(patch)
        pieces.append(
            {
                "color": color,
                "bbox": (box_h, box_w),
                "coords": _mask_coords(mask),
                "area": int(mask.sum()),
            }
        )

    outputs = _solve_outputs_from_pieces(pieces, box_h, box_w, limit=2)
    if len(outputs) != 1:
        return None
    out = outputs[0]

    random.shuffle(patches)
    inp = _scatter_patches(patches, bg, out.shape[0], out.shape[1])
    if inp is None:
        return None

    pieces, box_h2, box_w2 = _extract_pieces(inp)
    outputs = _solve_outputs_from_pieces(pieces, box_h2, box_w2, limit=2)
    if len(outputs) != 1 or not np.array_equal(outputs[0], out):
        return None

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _build_instance()
        if result is not None:
            return result[0]
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
