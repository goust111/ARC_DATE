import random
from collections import Counter, defaultdict

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BG = 0
GRID_H_RANGE = (16, 26)
GRID_W_RANGE = (16, 26)
FRAME_COUNT_CHOICES = (1, 2, 3)
FRAME_SIZE_RANGE = (6, 10)
FRAME_GAP = 2
MAX_TOTAL_CRACKS = 5
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _prefix_sum(mask: np.ndarray) -> np.ndarray:
    return mask.astype(int).cumsum(axis=0).cumsum(axis=1)


def _rect_sum(ps: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    if r0 > r1 or c0 > c1:
        return 0
    total = int(ps[r1, c1])
    if r0 > 0:
        total -= int(ps[r0 - 1, c1])
    if c0 > 0:
        total -= int(ps[r1, c0 - 1])
    if r0 > 0 and c0 > 0:
        total += int(ps[r0 - 1, c0 - 1])
    return total


def _perimeter_sum(ps: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    outer = _rect_sum(ps, r0, c0, r1, c1)
    inner = _rect_sum(ps, r0 + 1, c0 + 1, r1 - 1, c1 - 1)
    return outer - inner


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _ordered_line_signature(coords: list[tuple[int, int]]) -> tuple[tuple[str, int], tuple[tuple[int, int], ...]]:
    if len(coords) < 2:
        raise AssertionError("piece/crack length must be at least 2")
    rows = {r for r, _ in coords}
    cols = {c for _, c in coords}
    if len(rows) == 1:
        ordered = tuple(sorted(coords, key=lambda rc: rc[1]))
        if ordered[-1][1] - ordered[0][1] + 1 != len(ordered):
            raise AssertionError("horizontal line must be contiguous")
        return ("H", len(ordered)), ordered
    if len(cols) == 1:
        ordered = tuple(sorted(coords, key=lambda rc: rc[0]))
        if ordered[-1][0] - ordered[0][0] + 1 != len(ordered):
            raise AssertionError("vertical line must be contiguous")
        return ("V", len(ordered)), ordered
    raise AssertionError("component must be a straight line")


def _false_runs(bits: np.ndarray) -> list[tuple[int, int]]:
    runs = []
    start = None
    for idx, value in enumerate(bits.tolist()):
        if not value:
            if start is None:
                start = idx
        elif start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(bits) - 1))
    return runs


def _extract_cracks(frame_mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> list[dict]:
    cracks = []

    def add_horizontal(row: int) -> None:
        side = frame_mask[row, c0 + 1 : c1]
        for start, end in _false_runs(side):
            coords = [(row, c0 + 1 + k) for k in range(start, end + 1)]
            sig, ordered = _ordered_line_signature(coords)
            cracks.append({"sig": sig, "coords": ordered})

    def add_vertical(col: int) -> None:
        side = frame_mask[r0 + 1 : r1, col]
        for start, end in _false_runs(side):
            coords = [(r0 + 1 + k, col) for k in range(start, end + 1)]
            sig, ordered = _ordered_line_signature(coords)
            cracks.append({"sig": sig, "coords": ordered})

    add_horizontal(r0)
    add_horizontal(r1)
    add_vertical(c0)
    add_vertical(c1)
    return cracks


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _frame_candidates(
    grid: np.ndarray,
    bg: int,
    frame_color: int,
    piece_sigs: set[tuple[str, int]],
) -> tuple[list[dict], set[tuple[int, int]]]:
    h, w = grid.shape
    frame_mask = grid == frame_color
    non_bg = grid != bg
    invalid_mask = non_bg & ~frame_mask
    frame_ps = _prefix_sum(frame_mask)
    non_bg_ps = _prefix_sum(non_bg)
    invalid_ps = _prefix_sum(invalid_mask)

    frame_pixels = {tuple(map(int, rc)) for rc in np.argwhere(frame_mask)}
    rows = [int(r) for r in np.where(frame_mask.any(axis=1))[0]]
    cols = [int(c) for c in np.where(frame_mask.any(axis=0))[0]]

    candidates = []
    for i, r0 in enumerate(rows):
        for r1 in rows[i + 1 :]:
            if r1 - r0 < 5:
                continue
            for j, c0 in enumerate(cols):
                for c1 in cols[j + 1 :]:
                    if c1 - c0 < 5:
                        continue
                    if not (frame_mask[r0, c0] and frame_mask[r0, c1] and frame_mask[r1, c0] and frame_mask[r1, c1]):
                        continue
                    if _rect_sum(non_bg_ps, r0 + 1, c0 + 1, r1 - 1, c1 - 1) != 0:
                        continue
                    if _perimeter_sum(invalid_ps, r0, c0, r1, c1) != 0:
                        continue

                    cracks = _extract_cracks(frame_mask, r0, c0, r1, c1)
                    if not cracks:
                        continue
                    sigs = {crack["sig"] for crack in cracks}
                    if len(sigs) != len(cracks) or not sigs <= piece_sigs:
                        continue

                    occupied = set()
                    for c in range(c0, c1 + 1):
                        if frame_mask[r0, c]:
                            occupied.add((r0, c))
                        if frame_mask[r1, c]:
                            occupied.add((r1, c))
                    for r in range(r0 + 1, r1):
                        if frame_mask[r, c0]:
                            occupied.add((r, c0))
                        if frame_mask[r, c1]:
                            occupied.add((r, c1))

                    if len(occupied) <= 4:
                        continue
                    if not occupied <= frame_pixels:
                        continue

                    candidates.append(
                        {
                            "bbox": (r0, c0, r1, c1),
                            "frame_coords": occupied,
                            "cracks": cracks,
                            "sigs": sigs,
                            "area": (r1 - r0 + 1) * (c1 - c0 + 1),
                        }
                    )
    return candidates, frame_pixels


def _select_frames(candidates: list[dict], frame_pixels: set[tuple[int, int]], piece_sigs: set[tuple[str, int]]) -> list[dict]:
    pixel_to_candidates: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, cand in enumerate(candidates):
        for pixel in cand["frame_coords"]:
            pixel_to_candidates[pixel].append(idx)

    if not frame_pixels or not piece_sigs:
        raise AssertionError("need both frame pixels and crack signatures")
    if any(pixel not in pixel_to_candidates for pixel in frame_pixels):
        raise AssertionError("some frame pixels do not belong to any valid rectangle")

    def search(
        remaining_pixels: set[tuple[int, int]],
        remaining_sigs: set[tuple[str, int]],
        chosen_indices: tuple[int, ...],
    ) -> list[int] | None:
        if not remaining_pixels and not remaining_sigs:
            return []
        if not remaining_pixels or not remaining_sigs:
            return None

        chosen_boxes = [candidates[idx]["bbox"] for idx in chosen_indices]
        pivot = None
        pivot_options = None
        for pixel in remaining_pixels:
            options = []
            for idx in pixel_to_candidates[pixel]:
                cand = candidates[idx]
                if not cand["frame_coords"] <= remaining_pixels:
                    continue
                if not cand["sigs"] <= remaining_sigs:
                    continue
                if any(_bbox_overlap(cand["bbox"], box) for box in chosen_boxes):
                    continue
                options.append(idx)
            if not options:
                return None
            if pivot_options is None or len(options) < len(pivot_options):
                pivot = pixel
                pivot_options = options

        assert pivot is not None and pivot_options is not None
        pivot_options.sort(key=lambda idx: (-len(candidates[idx]["frame_coords"]), -len(candidates[idx]["sigs"]), -candidates[idx]["area"]))
        for idx in pivot_options:
            cand = candidates[idx]
            result = search(
                remaining_pixels - cand["frame_coords"],
                remaining_sigs - cand["sigs"],
                chosen_indices + (idx,),
            )
            if result is not None:
                return [idx] + result
        return None

    chosen = search(set(frame_pixels), set(piece_sigs), ())
    if chosen is None:
        raise AssertionError("failed to select a consistent set of frames")
    return [candidates[idx] for idx in chosen]


def _collect_pieces(grid: np.ndarray, bg: int, frame_color: int) -> dict[tuple[str, int], dict]:
    mask = (grid != bg) & (grid != frame_color)
    pieces = {}
    for comp in _components(mask):
        colors = {int(grid[r, c]) for r, c in comp}
        if len(colors) != 1:
            raise AssertionError("each external piece must be monochrome")
        sig, ordered = _ordered_line_signature(comp)
        if sig in pieces:
            raise AssertionError("piece signatures must be unique")
        pieces[sig] = {"color": colors.pop(), "coords": ordered}
    if not pieces:
        raise AssertionError("expected at least one external piece")
    return pieces


def _solve_with_frame_color(grid: np.ndarray, bg: int, frame_color: int) -> np.ndarray:
    pieces = _collect_pieces(grid, bg, frame_color)
    candidates, frame_pixels = _frame_candidates(grid, bg, frame_color, set(pieces))
    if not candidates:
        raise AssertionError("no frame candidates found")
    frames = _select_frames(candidates, frame_pixels, set(pieces))

    out = grid.copy()
    for piece in pieces.values():
        for r, c in piece["coords"]:
            out[r, c] = bg
    for frame in frames:
        for crack in frame["cracks"]:
            color = pieces[crack["sig"]]["color"]
            for r, c in crack["coords"]:
                out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    counts = Counter(int(v) for v in g.flat if int(v) != bg)
    if not counts:
        return g.copy()

    order = []
    seen = set()
    for value in g.flat:
        value = int(value)
        if value != bg and value not in seen:
            seen.add(value)
            order.append(value)
    order.sort(key=lambda color: counts[color], reverse=True)

    for frame_color in order:
        try:
            return _solve_with_frame_color(g, bg, frame_color)
        except AssertionError:
            continue
    raise AssertionError("no consistent interpretation found")


def _segment_cells(r0: int, c0: int, orient: str, length: int) -> list[tuple[int, int]]:
    if orient == "H":
        return [(r0, c0 + k) for k in range(length)]
    return [(r0 + k, c0) for k in range(length)]


def _dilate_cells(cells: list[tuple[int, int]], h: int, w: int, radius: int = 1) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _choose_frame_count(rng: random.Random) -> int:
    return rng.choice(FRAME_COUNT_CHOICES)


def _choose_cracks(
    rng: random.Random,
    height: int,
    width: int,
    used_sigs: set[tuple[str, int]],
    target_count: int,
) -> list[dict] | None:
    inner_h = height - 2
    inner_w = width - 2
    sides = ["top", "bottom", "left", "right"]
    rng.shuffle(sides)
    cracks = []

    for side in sides:
        if len(cracks) >= target_count:
            break
        orient = "H" if side in ("top", "bottom") else "V"
        max_len = inner_w if orient == "H" else inner_h
        if max_len < 2:
            continue
        upper = max_len - 1 if max_len >= 3 else max_len
        lengths = [length for length in range(2, upper + 1) if (orient, length) not in used_sigs]
        if not lengths:
            continue
        length = rng.choice(lengths)
        start = rng.randint(1, max_len - length + 1)
        crack = {"side": side, "orient": orient, "length": length, "start": start}
        cracks.append(crack)
        used_sigs.add((orient, length))

    return cracks if len(cracks) == target_count else None


def _frame_with_cracks(
    top: int,
    left: int,
    height: int,
    width: int,
    frame_color: int,
    cracks: list[dict],
) -> tuple[list[tuple[int, int]], list[dict]]:
    cells = []
    for c in range(left, left + width):
        cells.append((top, c))
        cells.append((top + height - 1, c))
    for r in range(top + 1, top + height - 1):
        cells.append((r, left))
        cells.append((r, left + width - 1))

    crack_specs = []
    removed = set()
    for crack in cracks:
        if crack["side"] == "top":
            coords = [(top, left + crack["start"] + offset) for offset in range(crack["length"])]
        elif crack["side"] == "bottom":
            coords = [(top + height - 1, left + crack["start"] + offset) for offset in range(crack["length"])]
        elif crack["side"] == "left":
            coords = [(top + crack["start"] + offset, left) for offset in range(crack["length"])]
        else:
            coords = [(top + crack["start"] + offset, left + width - 1) for offset in range(crack["length"])]
        sig, ordered = _ordered_line_signature(coords)
        removed.update(ordered)
        crack_specs.append({"sig": sig, "coords": ordered})

    frame_cells = [cell for cell in cells if cell not in removed]
    return frame_cells, crack_specs


def _place_frames(
    rng: random.Random,
    h: int,
    w: int,
    frame_color: int,
) -> tuple[np.ndarray, list[dict], list[tuple[str, int]]] | None:
    frame_count = _choose_frame_count(rng)
    grid = np.full((h, w), BG, dtype=int)
    forbidden: set[tuple[int, int]] = set()
    frames = []
    used_sigs: set[tuple[str, int]] = set()

    for frame_idx in range(frame_count):
        height = rng.randint(*FRAME_SIZE_RANGE)
        width = rng.randint(*FRAME_SIZE_RANGE)
        if height + 2 * FRAME_GAP > h or width + 2 * FRAME_GAP > w:
            return None

        if frame_count == 1:
            crack_target = rng.choice((1, 2, 3))
        else:
            crack_target = rng.choice((1, 2))
            if frame_idx == 0 and rng.random() < 0.3:
                crack_target = 3

        crack_target = min(crack_target, MAX_TOTAL_CRACKS - len(used_sigs))
        if crack_target <= 0:
            return None

        cracks = _choose_cracks(rng, height, width, used_sigs, crack_target)
        if cracks is None:
            return None

        placed = False
        for _ in range(120):
            top = rng.randint(FRAME_GAP, h - height - FRAME_GAP)
            left = rng.randint(FRAME_GAP, w - width - FRAME_GAP)
            bbox_cells = [(r, c) for r in range(top - FRAME_GAP, top + height + FRAME_GAP) for c in range(left - FRAME_GAP, left + width + FRAME_GAP)]
            if any((r, c) in forbidden for r, c in bbox_cells):
                continue

            frame_cells, crack_specs = _frame_with_cracks(top, left, height, width, frame_color, cracks)
            for r, c in frame_cells:
                grid[r, c] = frame_color
            forbidden.update((r, c) for r, c in bbox_cells if 0 <= r < h and 0 <= c < w)
            frames.append({"bbox": (top, left, top + height - 1, left + width - 1), "cracks": crack_specs})
            placed = True
            break
        if not placed:
            return None

    if not frames or len(used_sigs) > MAX_TOTAL_CRACKS:
        return None
    return grid, frames, list(used_sigs)


def _place_piece(
    rng: random.Random,
    grid: np.ndarray,
    forbidden: set[tuple[int, int]],
    orient: str,
    length: int,
    color: int,
) -> tuple[tuple[int, int], ...] | None:
    h, w = grid.shape
    for _ in range(200):
        if orient == "H":
            row = rng.randrange(h)
            col = rng.randrange(w - length + 1)
            cells = [(row, col + k) for k in range(length)]
        else:
            row = rng.randrange(h - length + 1)
            col = rng.randrange(w)
            cells = [(row + k, col) for k in range(length)]
        if any((r, c) in forbidden for r, c in cells):
            continue
        halo = _dilate_cells(cells, h, w, radius=1)
        if any((r, c) in forbidden for r, c in halo):
            continue
        for r, c in cells:
            if grid[r, c] != BG:
                break
        else:
            for r, c in cells:
                grid[r, c] = color
            forbidden.update(halo)
            return tuple(cells)
    return None


def _build_output(
    grid: np.ndarray,
    frames: list[dict],
    piece_map: dict[tuple[str, int], dict],
) -> np.ndarray:
    out = grid.copy()
    for piece in piece_map.values():
        for r, c in piece["cells"]:
            out[r, c] = BG
    for frame in frames:
        for crack in frame["cracks"]:
            color = piece_map[crack["sig"]]["color"]
            for r, c in crack["coords"]:
                out[r, c] = color
    return out


def _gen_attempt(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    palette = list(range(1, 10))
    frame_color = rng.choice(palette)
    palette.remove(frame_color)

    placed = _place_frames(rng, h, w, frame_color)
    if placed is None:
        return None
    grid, frames, sigs = placed
    if len(sigs) == 0 or len(sigs) > len(palette):
        return None

    forbidden = _dilate_cells(list(map(tuple, np.argwhere(grid != BG))), h, w, radius=1)
    rng.shuffle(palette)
    piece_map = {}
    for sig, color in zip(sigs, palette):
        cells = _place_piece(rng, grid, forbidden, sig[0], sig[1], color)
        if cells is None:
            return None
        piece_map[sig] = {"color": color, "cells": cells}

    expected = _build_output(grid, frames, piece_map)
    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        generated = _gen_attempt(rng)
        if generated is None:
            continue
        inp, expected = generated
        try:
            solved = main(inp)
        except AssertionError:
            continue
        if np.array_equal(solved, expected):
            return inp
    raise RuntimeError("failed to generate a valid sample")
