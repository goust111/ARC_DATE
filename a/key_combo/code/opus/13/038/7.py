from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7
PALETTE = [BLUE, GREEN, YELLOW, ORANGE]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

MAX_GENERATION_ATTEMPTS = 400
MAX_LOCAL_ATTEMPTS = 120

SHAPE_LIBRARY = {
    3: [
        np.array([[1, 0], [1, 1]], dtype=bool),
    ],
    4: [
        np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    ],
    5: [
        np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=bool),
    ],
    6: [
        np.array([[1, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=bool),
    ],
    7: [
        np.array([[1, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=bool),
    ],
    8: [
        np.array([[0, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool),
    ],
}


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_to_local(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    coords = np.argwhere(mask)
    bbox = _bbox_from_coords(coords)
    return _crop(mask.astype(int), bbox).astype(bool), bbox


def _object_components(grid: np.ndarray) -> list[dict]:
    comps = []
    for coords in _components(grid != BLACK):
        bbox = _bbox_from_coords(coords)
        patch = _crop(grid, bbox)
        rel = coords - np.array([bbox[0], bbox[1]])
        mask = np.zeros(patch.shape, dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True
        colors = set(map(int, patch[mask]))
        comps.append(
            {
                "coords": coords,
                "bbox": bbox,
                "patch": patch,
                "mask": mask,
                "colors": colors,
                "area": int(coords.shape[0]),
            }
        )
    return comps


def _hole_descriptors(frame_crop: np.ndarray) -> list[dict]:
    holes = []
    hole_mask = frame_crop == BLACK
    h, w = frame_crop.shape
    for coords in _components(hole_mask):
        rr, cc = coords[:, 0], coords[:, 1]
        if np.any(rr == 0) or np.any(rr == h - 1) or np.any(cc == 0) or np.any(cc == w - 1):
            continue
        bbox = _bbox_from_coords(coords)
        rel = coords - np.array([bbox[0], bbox[1]])
        local_mask = np.zeros((bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1), dtype=bool)
        local_mask[rel[:, 0], rel[:, 1]] = True
        holes.append(
            {
                "coords": coords,
                "coord_set": {tuple(map(int, rc)) for rc in coords},
                "bbox": bbox,
                "mask": local_mask,
                "area": int(coords.shape[0]),
            }
        )
    return holes


def _rotate_piece(piece: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(piece, k=int(k)).copy()


def _reconstruct_canvas(frame_crop: np.ndarray, piece_patches: list[np.ndarray]) -> np.ndarray:
    holes = _hole_descriptors(frame_crop)
    remaining = {idx: hole for idx, hole in enumerate(holes)}
    canvas = frame_crop.copy()

    pieces = sorted(piece_patches, key=lambda patch: int(np.count_nonzero(patch == RED)), reverse=True)
    assert len(pieces) == len(holes), "piece/hole count mismatch"

    for piece in pieces:
        candidates = []
        for rot in range(4):
            rotated = _rotate_piece(piece, rot)
            red_coords = np.argwhere(rotated == RED)
            assert red_coords.size > 0, "each piece needs a red stencil"
            for hole_id, hole in remaining.items():
                if int(red_coords.shape[0]) != hole["area"]:
                    continue
                shift = hole["coords"].min(axis=0) - red_coords.min(axis=0)
                shifted = {tuple(map(int, rc + shift)) for rc in red_coords}
                if shifted != hole["coord_set"]:
                    continue
                r0, c0 = map(int, shift)
                if r0 < 0 or c0 < 0:
                    continue
                if r0 + rotated.shape[0] > canvas.shape[0] or c0 + rotated.shape[1] > canvas.shape[1]:
                    continue
                candidates.append((hole_id, rotated, r0, c0))

        assert len(candidates) == 1, f"expected exactly one fit for a piece, found {len(candidates)}"
        hole_id, rotated, r0, c0 = candidates[0]
        for rr in range(rotated.shape[0]):
            for cc in range(rotated.shape[1]):
                value = int(rotated[rr, cc])
                if value == BLACK:
                    continue
                canvas[r0 + rr, c0 + cc] = RED if value == RED else value
        del remaining[hole_id]

    assert not remaining, "all holes must be filled"
    return canvas


def _unique_majority_color(canvas: np.ndarray) -> int:
    values = canvas[(canvas != BLACK) & (canvas != RED)]
    assert values.size > 0, "reconstructed canvas must contain non-red paint"
    colors, counts = np.unique(values, return_counts=True)
    max_count = int(counts.max())
    winners = colors[counts == max_count]
    assert winners.size == 1, "majority color must be unique"
    return int(winners[0])


def _parse_input(grid: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    comps = _object_components(grid[:-1, :])
    red_frames = [comp for comp in comps if comp["colors"] == {RED}]
    assert len(red_frames) == 1, "expected exactly one pure-red frame"
    frame = red_frames[0]
    frame_crop = frame["patch"]
    pieces = [comp["patch"] for comp in comps if comp is not frame]
    assert 2 <= len(pieces) <= 4, "expected 2-4 puzzle pieces"
    return frame_crop, pieces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the red canvas by fitting rotated puzzle stencils into its holes, then write the reconstructed majority color to the result bar.

    Input structure:
    - The grid has a black background and an all-black result bar in the bottom row.
    - Above the result bar there is one pure-red connected frame whose cropped bbox contains 2-4 enclosed black holes.
    - Separate 4-connected puzzle pieces sit on the background; each piece mixes red stencil cells with non-red paint cells.

    Rule:
    - Crop the pure-red frame to its bbox and detect the enclosed black hole components.
    - Sort pieces by red-stencil area descending, rotate each by quarter turns, and place it where its red cells exactly match one still-empty hole.
    - While placing, keep red stencil cells red and copy only non-red piece cells onto the cropped canvas; then count all non-black, non-red cells in the reconstructed canvas and paint the unique majority color into the bottom row's center column.

    Key insight:
    - The visible answer color depends on the fully reconstructed canvas, because later fitted pieces can change which non-red cells remain visible after all stencil-guided placements.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 2
    assert np.all(g[-1, :] == BLACK), "the input result bar must start black"

    frame_crop, pieces = _parse_input(g)
    canvas = _reconstruct_canvas(frame_crop, pieces)
    winner = _unique_majority_color(canvas)

    out = g.copy()
    out[-1, (g.shape[1] - 1) // 2] = winner
    return out


def _winner(counter: Counter[int]) -> int | None:
    if not counter:
        return None
    best = max(counter.values())
    winners = [color for color, count in counter.items() if count == best]
    return winners[0] if len(winners) == 1 else None


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _shape_candidates(area: int) -> list[np.ndarray]:
    shapes = []
    for base in SHAPE_LIBRARY[area]:
        for k in range(4):
            shape = _rotate_piece(base.astype(int), k).astype(bool)
            if any(np.array_equal(shape, _rotate_piece(shape.astype(int), rk).astype(bool)) for rk in (1, 2, 3)):
                continue
            key = (shape.shape, shape.tobytes())
            if all((other.shape, other.tobytes()) != key for other in shapes):
                shapes.append(shape)
    return shapes


def _place_holes(frame_h: int, frame_w: int, areas: list[int]) -> list[dict] | None:
    occupied = np.zeros((frame_h, frame_w), dtype=bool)
    blocked = np.zeros((frame_h, frame_w), dtype=bool)
    holes: list[dict] = []
    center = np.array([frame_h / 2.0, frame_w / 2.0])

    for area in areas:
        choices = []
        for shape in _shape_candidates(area):
            sh, sw = shape.shape
            for r0 in range(1, frame_h - sh):
                for c0 in range(1, frame_w - sw):
                    if blocked[r0 : r0 + sh, c0 : c0 + sw][shape].any():
                        continue
                    coords = np.argwhere(shape) + np.array([r0, c0])
                    score = float(np.abs(coords.mean(axis=0) - center).sum())
                    if holes:
                        dists = [
                            float(np.abs(coords.mean(axis=0) - hole["coords"].mean(axis=0)).sum())
                            for hole in holes
                        ]
                        score += min(dists) * 0.35
                    choices.append((score, random.random(), shape, r0, c0))
        if not choices:
            return None
        choices.sort(key=lambda item: (item[0], item[1]))
        pool = choices[: max(1, len(choices) // 3)]
        _score, _noise, shape, r0, c0 = random.choice(pool)
        sh, sw = shape.shape
        local_coords = np.argwhere(shape)
        coords = local_coords + np.array([r0, c0])
        hole_mask = np.zeros((frame_h, frame_w), dtype=bool)
        hole_mask[coords[:, 0], coords[:, 1]] = True
        occupied |= hole_mask
        blocked = _dilate4(occupied)
        holes.append(
            {
                "area": area,
                "shape": shape,
                "coords": coords,
                "bbox": (r0, c0, r0 + sh - 1, c0 + sw - 1),
            }
        )

    for i in range(len(holes)):
        for j in range(i + 1, len(holes)):
            a, b = holes[i]["bbox"], holes[j]["bbox"]
            row_gap = max(0, max(a[0], b[0]) - min(a[2], b[2]) - 1)
            col_gap = max(0, max(a[1], b[1]) - min(a[3], b[3]) - 1)
            if row_gap + col_gap <= 2:
                return holes
    return None


def _pick_margin(available: int, prefer_large: bool) -> int:
    if available <= 0:
        return 0
    if available == 1:
        return 1
    if prefer_large:
        return 2 if random.random() < 0.7 else 1
    return 1 if random.random() < 0.7 else 2


def _build_piece_specs(frame_shape: tuple[int, int], holes: list[dict]) -> list[dict] | None:
    frame_h, frame_w = frame_shape
    pieces = []
    for index, hole in enumerate(holes):
        r0, c0, r1, c1 = hole["bbox"]
        hole_h = r1 - r0 + 1
        hole_w = c1 - c0 + 1
        prefer_large = index > 0
        top = _pick_margin(r0, prefer_large)
        left = _pick_margin(c0, prefer_large)
        bottom = _pick_margin(frame_h - r1 - 1, prefer_large)
        right = _pick_margin(frame_w - c1 - 1, prefer_large)
        patch_h = hole_h + top + bottom
        patch_w = hole_w + left + right
        patch = np.zeros((patch_h, patch_w), dtype=int)
        patch[:, :] = BLUE
        red_rows, red_cols = np.where(hole["shape"])
        patch[red_rows + top, red_cols + left] = RED
        pieces.append(
            {
                "area": hole["area"],
                "origin": (r0 - top, c0 - left),
                "patch": patch,
                "visible_cells": int(patch_h * patch_w - hole["area"]),
            }
        )

    overlap_pairs = 0
    for i in range(len(pieces)):
        ai0, aj0 = pieces[i]["origin"]
        ah, aw = pieces[i]["patch"].shape
        ai1, aj1 = ai0 + ah - 1, aj0 + aw - 1
        for j in range(i + 1, len(pieces)):
            bi0, bj0 = pieces[j]["origin"]
            bh, bw = pieces[j]["patch"].shape
            bi1, bj1 = bi0 + bh - 1, bj0 + bw - 1
            ir0, ic0 = max(ai0, bi0), max(aj0, bj0)
            ir1, ic1 = min(ai1, bi1), min(aj1, bj1)
            if ir0 <= ir1 and ic0 <= ic1:
                overlap_pairs += 1
    return pieces if overlap_pairs > 0 else None


def _simulate_piece_majority(frame_crop: np.ndarray, pieces: list[dict]) -> tuple[Counter[int], Counter[int], np.ndarray]:
    visible_counts: Counter[int] = Counter()
    canvas = frame_crop.copy()

    for piece in pieces:
        visible_counts[int(piece["color"])] += int(np.count_nonzero(piece["patch"] != RED))

    for piece in sorted(pieces, key=lambda item: item["area"], reverse=True):
        r0, c0 = piece["origin"]
        patch = piece["patch"]
        for rr in range(patch.shape[0]):
            for cc in range(patch.shape[1]):
                value = int(patch[rr, cc])
                if value == RED:
                    canvas[r0 + rr, c0 + cc] = RED
                elif value != BLACK:
                    canvas[r0 + rr, c0 + cc] = value

    final_counts = Counter(map(int, canvas[(canvas != BLACK) & (canvas != RED)]))
    return visible_counts, final_counts, canvas


def _construct_local_instance() -> tuple[np.ndarray, list[np.ndarray], int] | None:
    n_pieces = random.choice([2, 3, 3, 4, 4])
    frame_h = random.randint(max(9, 6 + n_pieces), 14)
    frame_w = random.randint(max(9, 6 + n_pieces), 14)
    areas = sorted(random.sample(list(SHAPE_LIBRARY), n_pieces), reverse=True)

    holes = _place_holes(frame_h, frame_w, areas)
    if holes is None:
        return None

    pieces = _build_piece_specs((frame_h, frame_w), holes)
    if pieces is None:
        return None

    colors = random.sample(PALETTE, n_pieces)
    for piece, color in zip(pieces, colors):
        piece["patch"] = np.where(piece["patch"] == BLUE, color, piece["patch"]).astype(int)
        piece["color"] = int(color)

    frame_crop = np.full((frame_h, frame_w), RED, dtype=int)
    for hole in holes:
        coords = hole["coords"]
        frame_crop[coords[:, 0], coords[:, 1]] = BLACK

    if len(_components(frame_crop == RED)) != 1:
        return None
    if len(_hole_descriptors(frame_crop)) != n_pieces:
        return None

    visible_counts, final_counts, canvas = _simulate_piece_majority(frame_crop, pieces)
    visible_winner = _winner(visible_counts)
    final_winner = _winner(final_counts)
    if visible_winner is None or final_winner is None or visible_winner == final_winner:
        return None

    if np.count_nonzero(canvas == final_winner) < 4:
        return None

    input_pieces = []
    rotations = [random.randint(0, 3) for _ in pieces]
    if all(rot == 0 for rot in rotations):
        rotations[random.randrange(len(rotations))] = random.choice([1, 2, 3])
    for piece, rot in zip(pieces, rotations):
        input_pieces.append(_rotate_piece(piece["patch"], rot))

    return frame_crop, input_pieces, int(final_winner)


def _can_place(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, pad: int = 1) -> bool:
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > dst.shape[0] or c0 + w > dst.shape[1]:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(dst.shape[0], r0 + h + pad)
    cc1 = min(dst.shape[1], c0 + w + pad)
    return not np.any(dst[rr0:rr1, cc0:cc1] != BLACK)


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BLACK
    view[mask] = patch[mask]


def _scatter_into_full_grid(frame_crop: np.ndarray, pieces: list[np.ndarray], winner: int) -> np.ndarray | None:
    frame_h, frame_w = frame_crop.shape
    piece_heights = [piece.shape[0] for piece in pieces]
    piece_widths = [piece.shape[1] for piece in pieces]
    min_h = max(frame_h + 2, max(piece_heights) + 2, 12) + 1
    min_w = max(frame_w + 2, max(piece_widths) + 2, 14)
    need_rows = frame_h + sum(piece_heights) + len(pieces) + 4
    need_cols = frame_w + max(piece_widths) + 4
    min_h = max(min_h, min(29, need_rows // 2 + 4))
    min_w = max(min_w, min(30, need_cols))
    full_h = random.randint(min_h, min(29, max(min_h, 22)))
    full_w = random.randint(min_w, min(30, max(min_w, 24)))
    grid = np.zeros((full_h + 1, full_w), dtype=int)

    frame_positions = [
        (r, c)
        for r in range(1, full_h - frame_h)
        for c in range(0, full_w - frame_w + 1)
    ]
    random.shuffle(frame_positions)
    frame_r, frame_c = None, None
    for r, c in frame_positions:
        if _can_place(grid[:-1, :], frame_crop, r, c, pad=1):
            frame_r, frame_c = r, c
            break
    if frame_r is None:
        return None
    _paste(grid[:-1, :], frame_crop, frame_r, frame_c)

    order = list(range(len(pieces)))
    random.shuffle(order)
    for idx in order:
        piece = pieces[idx]
        candidates = [
            (r, c)
            for r in range(0, full_h - piece.shape[0] + 1)
            for c in range(0, full_w - piece.shape[1] + 1)
            if _can_place(grid[:-1, :], piece, r, c, pad=1)
        ]
        if not candidates:
            return None
        r, c = random.choice(candidates)
        _paste(grid[:-1, :], piece, r, c)

    assert np.all(grid[-1, :] == BLACK)
    expected = grid.copy()
    expected[-1, (grid.shape[1] - 1) // 2] = winner
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        frame_crop, pieces = _parse_input(grid)
        holes = _hole_descriptors(frame_crop)
        if not (2 <= len(holes) <= 4 and len(holes) == len(pieces)):
            return False
        canvas = _reconstruct_canvas(frame_crop, pieces)
        final_values = canvas[(canvas != BLACK) & (canvas != RED)]
        if final_values.size == 0:
            return False
        final_counts = Counter(map(int, final_values))
        final_winner = _winner(final_counts)
        if final_winner is None:
            return False

        visible_counts: Counter[int] = Counter()
        for piece in pieces:
            values = piece[(piece != BLACK) & (piece != RED)]
            piece_counts = Counter(map(int, values))
            visible_counts.update(piece_counts)
        visible_winner = _winner(visible_counts)
        if visible_winner is None or visible_winner == final_winner:
            return False

        out = main(grid)
        if out[-1, (grid.shape[1] - 1) // 2] != final_winner:
            return False
        if not np.array_equal(out[:-1, :], grid[:-1, :]):
            return False
        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        local = _construct_local_instance()
        if local is None:
            continue
        frame_crop, pieces, winner = local
        for _ in range(MAX_LOCAL_ATTEMPTS):
            grid = _scatter_into_full_grid(frame_crop, pieces, winner)
            if grid is not None and _validate_instance(grid):
                return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
