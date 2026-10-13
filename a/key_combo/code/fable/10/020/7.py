from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
ORIENTATION_NAMES = ("identity", "hflip", "vflip", "hvflip")
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_non_bg_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coord_arr = np.array(coords, dtype=int)
            components.append(
                {
                    "coords": coord_arr,
                    "bbox": _bbox_from_coords(coord_arr),
                    "area": int(coord_arr.shape[0]),
                }
            )
    return components


def _border_signature(patch: np.ndarray, bg: int) -> tuple[int, int, int, int] | None:
    h, w = patch.shape
    if h < 3 or w < 3:
        return None

    top = patch[0, :]
    bottom = patch[-1, :]
    left = patch[1:-1, 0]
    right = patch[1:-1, -1]

    sig_parts = (top, bottom, left, right)
    colors: list[int] = []
    for part in sig_parts:
        vals = set(map(int, part))
        if len(vals) != 1:
            return None
        color = next(iter(vals))
        if color == bg:
            return None
        colors.append(color)
    return tuple(colors)  # type: ignore[return-value]


def _apply_orientation(patch: np.ndarray, orientation: int) -> np.ndarray:
    if orientation == 0:
        return patch.copy()
    if orientation == 1:
        return np.fliplr(patch)
    if orientation == 2:
        return np.flipud(patch)
    if orientation == 3:
        return np.flipud(np.fliplr(patch))
    raise ValueError(f"bad orientation {orientation}")


def _frame_patch(height: int, width: int, sig: tuple[int, int, int, int]) -> np.ndarray:
    top, bottom, left, right = sig
    patch = np.full((height, width), BG, dtype=int)
    patch[0, :] = top
    patch[-1, :] = bottom
    patch[1:-1, 0] = left
    patch[1:-1, -1] = right
    return patch


def _build_output(solution: np.ndarray, sig: tuple[int, int, int, int]) -> np.ndarray:
    h, w = solution.shape
    out = _frame_patch(h + 2, w + 2, sig)
    out[1:-1, 1:-1] = solution
    return out


def _evaluate_candidates(
    candidates: list[np.ndarray],
    frame_sig: tuple[int, int, int, int],
    residue_mask: np.ndarray,
    residue_vals: np.ndarray,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for idx, candidate in enumerate(candidates):
        for orientation in range(4):
            oriented = _apply_orientation(candidate, orientation)
            sig = _border_signature(oriented, BG)
            if sig is None:
                continue
            edge_ok = sig == frame_sig
            residue_ok = bool(np.array_equal(oriented[residue_mask], residue_vals))
            results.append(
                {
                    "candidate_idx": idx,
                    "orientation": orientation,
                    "oriented": oriented,
                    "edge_ok": edge_ok,
                    "residue_ok": residue_ok,
                }
            )
    return results


def _find_frame(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    candidates: list[tuple[int, tuple[int, int, int, int], tuple[int, int, int, int]]] = []
    for comp in _find_non_bg_components(grid, bg):
        bbox = comp["bbox"]
        patch = _crop_bbox(grid, bbox)
        sig = _border_signature(patch, bg)
        if sig is None:
            continue
        interior = patch[1:-1, 1:-1]
        if interior.size == 0:
            continue
        if not np.any(interior == bg):
            continue
        if not np.any(interior != bg):
            continue
        area = patch.shape[0] * patch.shape[1]
        candidates.append((area, bbox, sig))

    assert candidates, "expected a framed lock rectangle"
    max_area = max(area for area, _bbox, _sig in candidates)
    largest = [(bbox, sig) for area, bbox, sig in candidates if area == max_area]
    assert len(largest) == 1, "expected a unique largest lock frame"
    return largest[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the lock cavity with the only candidate motif whose mirrored edge colors and residue overlap both fit.

    Input structure:
    - A single rectangular lock frame appears on the canvas with 1-cell-thick top,
      bottom, left, and right bars; each side bar is monochrome, and the corners
      belong to the top/bottom bars.
    - The frame interior is mostly background, but a few colored residue cells
      remain inside it as partial evidence of the missing key.
    - Elsewhere on the canvas are several separated candidate motif blocks with the
      same size as the frame interior; each candidate has a monochrome top edge,
      bottom edge, left edge, and right edge.

    Rule:
    - Detect the unique lock frame and crop the final output to that frame.
    - For every candidate block, test the four allowed mirror states: identity,
      horizontal flip, vertical flip, and both flips.
    - Keep only orientations whose four border colors match the frame sides, then
      require every residue cell inside the frame to match the candidate cell at the
      same interior coordinate.
    - Exactly one candidate orientation passes both checks; place that full oriented
      motif into the frame interior and return the cropped framed result.

    Key insight:
    - Border colors alone do not identify the right motif, and residues alone do not
      identify the right orientation; the answer is the unique motif that satisfies
      both locks simultaneously.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frame_bbox, frame_sig = _find_frame(g, bg)
    r0, c0, r1, c1 = frame_bbox
    cavity = g[r0 + 1 : r1, c0 + 1 : c1]
    residue_mask = cavity != bg
    residue_vals = cavity[residue_mask]

    cavity_shape = cavity.shape
    candidates: list[np.ndarray] = []
    for comp in _find_non_bg_components(g, bg):
        bbox = comp["bbox"]
        if bbox == frame_bbox:
            continue
        patch = _crop_bbox(g, bbox)
        if patch.shape != cavity_shape:
            continue
        if _border_signature(patch, bg) is None:
            continue
        candidates.append(patch)

    assert candidates, "expected at least one candidate block"
    results = _evaluate_candidates(candidates, frame_sig, residue_mask, residue_vals)
    matches = [rec for rec in results if rec["edge_ok"] and rec["residue_ok"]]
    assert len(matches) == 1, "expected a unique candidate/orientation match"

    solved = np.array(matches[0]["oriented"], dtype=int)
    out = _crop_bbox(g, frame_bbox)
    out[1:-1, 1:-1] = solved
    return out


def _sample_frame_signature() -> tuple[int, int, int, int]:
    palette = list(range(1, 10))
    mode = random.random()
    if mode < 0.45:
        return tuple(random.sample(palette, 4))  # type: ignore[return-value]
    if mode < 0.75:
        a, b, c = random.sample(palette, 3)
        pattern = random.choice(
            [
                (a, a, b, c),
                (a, b, c, c),
                (a, b, a, c),
                (a, b, c, a),
            ]
        )
        return tuple(map(int, pattern))
    a, b = random.sample(palette, 2)
    pattern = random.choice(
        [
            (a, b, a, b),
            (a, a, b, b),
            (a, b, b, a),
        ]
    )
    return tuple(map(int, pattern))


def _random_interior_palette(sig: tuple[int, int, int, int]) -> list[int]:
    available = list(range(1, 10))
    k = random.randint(3, 5)
    picked = random.sample(available, k)
    for color in sig:
        if color not in picked and random.random() < 0.45:
            picked[random.randrange(len(picked))] = color
    return sorted(set(map(int, picked)))


def _build_motif(height: int, width: int, sig: tuple[int, int, int, int]) -> np.ndarray | None:
    motif = _frame_patch(height, width, sig)
    palette = _random_interior_palette(sig)
    interior = motif[1:-1, 1:-1]
    for r in range(interior.shape[0]):
        for c in range(interior.shape[1]):
            interior[r, c] = random.choice(palette)
    if len(set(map(int, interior.flatten()))) < 2:
        return None
    views = [_apply_orientation(motif, ori) for ori in range(4)]
    if len({view.tobytes() for view in views}) < 3:
        return None
    return motif


def _choose_residue_coords(solution: np.ndarray) -> list[tuple[int, int]] | None:
    inner_h, inner_w = solution.shape[0] - 2, solution.shape[1] - 2
    coords = [(r, c) for r in range(1, inner_h + 1) for c in range(1, inner_w + 1)]
    if len(coords) < 4:
        return None
    lo = max(3, len(coords) // 4)
    hi = max(lo, (len(coords) * 3) // 5)
    count = random.randint(lo, hi)
    picked = random.sample(coords, count)
    residue_colors = {int(solution[r, c]) for r, c in picked}
    if len(residue_colors) < 2:
        return None
    return picked


def _mutate_edge_decoy(solution: np.ndarray, residues: list[tuple[int, int]]) -> np.ndarray:
    decoy = solution.copy()
    choices = list(residues)
    random.shuffle(choices)
    n_changes = random.randint(1, max(1, min(3, len(choices))))
    palette = [c for c in range(1, 10)]
    for r, c in choices[:n_changes]:
        new_color = random.choice([col for col in palette if col != int(solution[r, c])])
        decoy[r, c] = new_color
    return decoy


def _sample_wrong_signature(frame_sig: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    palette = list(range(1, 10))
    top_bottom = {frame_sig[0], frame_sig[1]}
    left_right = {frame_sig[2], frame_sig[3]}
    while True:
        sig = _sample_frame_signature()
        if set(sig[:2]) != top_bottom or set(sig[2:]) != left_right:
            return sig
        for idx in random.sample(range(4), 4):
            replacement = random.choice([c for c in palette if c != frame_sig[idx]])
            trial = list(sig)
            trial[idx] = replacement
            trial_sig = tuple(trial)
            if set(trial_sig[:2]) != top_bottom or set(trial_sig[2:]) != left_right:
                return trial_sig


def _build_residue_decoy(
    solution: np.ndarray,
    residues: list[tuple[int, int]],
    frame_sig: tuple[int, int, int, int],
) -> np.ndarray | None:
    wrong_sig = _sample_wrong_signature(frame_sig)
    decoy = _build_motif(solution.shape[0], solution.shape[1], wrong_sig)
    if decoy is None:
        return None
    for r, c in residues:
        decoy[r, c] = solution[r, c]
    return decoy


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _place_boxes(
    canvas_shape: tuple[int, int],
    sizes: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    height, width = canvas_shape
    placed_boxes: list[tuple[int, int, int, int]] = []
    anchors: list[tuple[int, int]] = []
    order = list(range(len(sizes)))
    random.shuffle(order)
    by_index: dict[int, tuple[int, int]] = {}

    for idx in order:
        h, w = sizes[idx]
        positions = [
            (r, c)
            for r in range(height - h + 1)
            for c in range(width - w + 1)
            if all(_boxes_separated((r, c, r + h - 1, c + w - 1), box) for box in placed_boxes)
        ]
        if not positions:
            return None
        r, c = random.choice(positions)
        placed_boxes.append((r, c, r + h - 1, c + w - 1))
        by_index[idx] = (r, c)

    for idx in range(len(sizes)):
        anchors.append(by_index[idx])
    return anchors


def _build_expected_input(
    solution: np.ndarray,
    solution_orientation: int,
    edge_decoy: np.ndarray,
    edge_orientation: int,
    residue_decoy: np.ndarray,
    residue_orientation: int,
    extras: list[tuple[np.ndarray, int]],
    frame_sig: tuple[int, int, int, int],
    residues: list[tuple[int, int]],
    canvas_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    h, w = solution.shape
    frame = _frame_patch(h + 2, w + 2, frame_sig)
    for r, c in residues:
        frame[r + 1, c + 1] = solution[r, c]

    raw_candidates = [
        _apply_orientation(solution, solution_orientation),
        _apply_orientation(edge_decoy, edge_orientation),
        _apply_orientation(residue_decoy, residue_orientation),
    ]
    raw_candidates.extend(_apply_orientation(patch, orientation) for patch, orientation in extras)

    sizes = [(h + 2, w + 2)] + [patch.shape for patch in raw_candidates]
    anchors = _place_boxes(canvas_shape, sizes)
    if anchors is None:
        raise ValueError("placement failed")

    grid = np.full(canvas_shape, BG, dtype=int)
    frame_r, frame_c = anchors[0]
    _paste(grid, frame, frame_r, frame_c)
    for (r, c), patch in zip(anchors[1:], raw_candidates):
        _paste(grid, patch, r, c)

    expected = _build_output(solution, frame_sig)
    return grid, expected, raw_candidates


def _semantic_checks(
    raw_candidates: list[np.ndarray],
    solution: np.ndarray,
    frame_sig: tuple[int, int, int, int],
    residues: list[tuple[int, int]],
) -> bool:
    residue_mask = np.zeros(solution.shape, dtype=bool)
    for r, c in residues:
        residue_mask[r, c] = True
    residue_vals = solution[residue_mask]
    results = _evaluate_candidates(raw_candidates, frame_sig, residue_mask, residue_vals)
    dual = [rec for rec in results if rec["edge_ok"] and rec["residue_ok"]]
    edge_only = [rec for rec in results if rec["edge_ok"] and not rec["residue_ok"]]
    residue_only = [rec for rec in results if rec["residue_ok"] and not rec["edge_ok"]]
    if len(dual) != 1:
        return False
    if not edge_only or not residue_only:
        return False
    winning = dual[0]
    if int(winning["candidate_idx"]) != 0:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    cavity_h = random.randint(5, 8)
    cavity_w = random.randint(5, 8)
    frame_sig = _sample_frame_signature()

    solution = _build_motif(cavity_h, cavity_w, frame_sig)
    if solution is None:
        return None
    residues = _choose_residue_coords(solution)
    if residues is None:
        return None

    edge_decoy = _mutate_edge_decoy(solution, residues)
    residue_decoy = _build_residue_decoy(solution, residues, frame_sig)
    if residue_decoy is None:
        return None

    solution_orientation = random.randrange(4)
    edge_orientation = random.randrange(4)
    residue_orientation = random.randrange(4)

    extras: list[tuple[np.ndarray, int]] = []
    for _ in range(random.randint(0, 2)):
        if random.random() < 0.5:
            patch = _mutate_edge_decoy(solution, residues)
        else:
            patch = _build_residue_decoy(solution, residues, frame_sig)
            if patch is None:
                return None
        extras.append((patch, random.randrange(4)))

    canvas_h = random.randint(max(cavity_h + 10, 18), 30)
    canvas_w = random.randint(max(cavity_w + 10, 18), 30)
    try:
        grid, expected, raw_candidates = _build_expected_input(
            solution,
            solution_orientation,
            edge_decoy,
            edge_orientation,
            residue_decoy,
            residue_orientation,
            extras,
            frame_sig,
            residues,
            (canvas_h, canvas_w),
        )
    except ValueError:
        return None

    if not _semantic_checks(raw_candidates, solution, frame_sig, residues):
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
