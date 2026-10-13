from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_TOOLBOX_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_TOOLBOX_DIR) not in sys.path:
    sys.path.append(str(_TOOLBOX_DIR))

from arc_utils import Color, bbox_coords, neighbors8


INPUT_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
OUTPUT_COLOR = Color.RED
GEN_ATTEMPTS = 500


def _prefix_sum(mask: np.ndarray) -> np.ndarray:
    return np.pad(mask.astype(np.int16), ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)


def _rect_sum(prefix: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    return int(prefix[r1 + 1, c1 + 1] - prefix[r0, c1 + 1] - prefix[r1 + 1, c0] + prefix[r0, c0])


def _majority_color(values: np.ndarray) -> int:
    counts = np.bincount(values.astype(int), minlength=12)
    return int(np.argmax(counts))


def _expand_tile(tile: np.ndarray, shape: tuple[int, int], shift: tuple[int, int] = (0, 0)) -> np.ndarray:
    pr, pc = tile.shape
    sr, sc = shift
    rows = (np.arange(shape[0]) + int(sr)) % pr
    cols = (np.arange(shape[1]) + int(sc)) % pc
    return tile[np.ix_(rows, cols)]


def _axis_shift_score(grid: np.ndarray, axis: int, shift: int) -> float:
    if axis == 0:
        a = grid[:-shift, :]
        b = grid[shift:, :]
    else:
        a = grid[:, :-shift]
        b = grid[:, shift:]
    return float(np.count_nonzero(a == b)) / float(a.size)


def _detect_axis_period(grid: np.ndarray, axis: int) -> int:
    limit = min(8, grid.shape[axis] - 1)
    scores = [(shift, _axis_shift_score(grid, axis, shift)) for shift in range(1, limit + 1)]
    best_score = max(score for _shift, score in scores)
    near_best = [shift for shift, score in scores if score >= best_score - 0.02]
    return min(near_best)


def _reconstruct_tile(grid: np.ndarray, pr: int, pc: int) -> np.ndarray:
    tile = np.zeros((pr, pc), dtype=int)
    for rr in range(pr):
        for cc in range(pc):
            tile[rr, cc] = _majority_color(grid[rr::pr, cc::pc].ravel())
    return tile


def _rank_shift_phases(grid: np.ndarray, base: np.ndarray, tile: np.ndarray) -> list[tuple[int, int, int]]:
    mismatch = grid != base
    ranked: list[tuple[int, int, int]] = []
    pr, pc = tile.shape
    for sr in range(pr):
        for sc in range(pc):
            if sr == 0 and sc == 0:
                continue
            shifted = _expand_tile(tile, grid.shape, (sr, sc))
            votes = int(np.count_nonzero(mismatch & (grid == shifted)))
            if votes > 0:
                ranked.append((sr, sc, votes))
    ranked.sort(key=lambda item: item[2], reverse=True)
    return ranked


def _best_rectangle_for_shift(
    grid: np.ndarray, base: np.ndarray, tile: np.ndarray, shift: tuple[int, int]
) -> dict | None:
    shifted = _expand_tile(tile, grid.shape, shift)
    changed = shifted != base
    visible = (grid == shifted) & changed
    visible_count = int(visible.sum())
    if visible_count == 0:
        return None

    visible_coords = np.argwhere(visible)
    _visible_bbox = bbox_coords(visible_coords)
    rows = np.unique(visible_coords[:, 0])
    cols = np.unique(visible_coords[:, 1])
    prefix_visible = _prefix_sum(visible)
    prefix_changed = _prefix_sum(changed)

    best: dict | None = None
    best_score: tuple | None = None
    for r0 in rows:
        for r1 in rows[rows >= r0]:
            for c0 in cols:
                for c1 in cols[cols >= c0]:
                    inside_visible = _rect_sum(prefix_visible, int(r0), int(c0), int(r1), int(c1))
                    if inside_visible == 0:
                        continue
                    inside_changed = _rect_sum(prefix_changed, int(r0), int(c0), int(r1), int(c1))
                    if inside_visible != inside_changed:
                        continue
                    if _rect_sum(prefix_visible, int(r0), int(c0), int(r0), int(c1)) == 0:
                        continue
                    if _rect_sum(prefix_visible, int(r1), int(c0), int(r1), int(c1)) == 0:
                        continue
                    if _rect_sum(prefix_visible, int(r0), int(c0), int(r1), int(c0)) == 0:
                        continue
                    if _rect_sum(prefix_visible, int(r0), int(c1), int(r1), int(c1)) == 0:
                        continue

                    area = (int(r1) - int(r0) + 1) * (int(c1) - int(c0) + 1)
                    outside_visible = visible_count - inside_visible
                    score = (outside_visible == 0, inside_visible, -outside_visible, -area)
                    if best_score is None or score > best_score:
                        best_score = score
                        best = {
                            "rect": (int(r0), int(c0), int(r1), int(c1)),
                            "support": inside_visible,
                            "outside_visible": outside_visible,
                            "area": area,
                            "score": score,
                        }
    return best


def _solve_details(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    pr = _detect_axis_period(g, axis=0)
    pc = _detect_axis_period(g, axis=1)
    tile = _reconstruct_tile(g, pr, pc)
    base = _expand_tile(tile, g.shape)

    ranked_shifts = _rank_shift_phases(g, base, tile)
    assert ranked_shifts, "expected at least one plausible phase shift"

    candidates = []
    for sr, sc, votes in ranked_shifts[: min(6, len(ranked_shifts))]:
        rect_info = _best_rectangle_for_shift(g, base, tile, (sr, sc))
        if rect_info is None:
            continue
        overall_score = (
            rect_info["outside_visible"] == 0,
            rect_info["support"],
            -rect_info["outside_visible"],
            -rect_info["area"],
            votes,
        )
        candidates.append(
            {
                "shift": (sr, sc),
                "votes": votes,
                "rect": rect_info["rect"],
                "support": rect_info["support"],
                "outside_visible": rect_info["outside_visible"],
                "area": rect_info["area"],
                "score": overall_score,
                "tile": tile,
                "base": base,
                "period": (pr, pc),
            }
        )

    assert candidates, "expected at least one coherent rectangle candidate"
    candidates.sort(key=lambda item: item["score"], reverse=True)
    best = candidates[0]
    best["runner_up_score"] = candidates[1]["score"] if len(candidates) > 1 else None
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect the phase-shifted rectangle in a periodic pattern and repaint the whole domain red.

    Input structure:
    - The entire canvas is filled by a two-dimensional repeating multicolor pattern.
    - One solid axis-aligned rectangle contains the same pattern under one non-zero phase shift.
    - A few isolated noise pixels may violate the pattern but do not form any coherent structure.

    Rule:
    - Infer the global row and column periods from the near-perfect translational symmetries.
    - Reconstruct the canonical periodic background, then test which non-zero phase shift explains a solid rectangle of mismatching cells.
    - Recolor every cell of that rectangle red, including hidden members whose shifted color happens to match the original pattern, while leaving isolated noise unchanged.

    Key insight:
    - The visible mismatches are only the changed residues of one shifted pattern; fitting them to a single phase vector and a full rectangle recovers the invisible matching cells and rejects isolated noise.
    """
    g = np.asarray(grid, dtype=int)
    info = _solve_details(g)
    r0, c0, r1, c1 = info["rect"]
    out = g.copy()
    out[r0 : r1 + 1, c0 : c1 + 1] = OUTPUT_COLOR
    return out


def _tile_has_period(tile: np.ndarray, axis: int, shift: int) -> bool:
    return bool(np.array_equal(tile, np.roll(tile, -int(shift), axis=axis)))


def _sample_tile(pr: int, pc: int) -> np.ndarray:
    for _ in range(300):
        palette = random.sample(INPUT_COLORS, random.randint(3, min(5, len(INPUT_COLORS))))
        tile = np.array([[random.choice(palette) for _ in range(pc)] for _ in range(pr)], dtype=int)
        if len(set(map(int, tile.ravel()))) < 3:
            continue
        if any(_tile_has_period(tile, 0, shift) for shift in range(1, pr)):
            continue
        if any(_tile_has_period(tile, 1, shift) for shift in range(1, pc)):
            continue
        return tile
    raise RuntimeError("failed to sample a primitive tile")


def _shift_modes(pr: int, pc: int) -> list[tuple[int, int]]:
    phases = [(sr, sc) for sr in range(pr) for sc in range(pc) if not (sr == 0 and sc == 0)]
    modes = []
    for want in ("row", "col", "both"):
        if want == "row":
            options = [phase for phase in phases if phase[0] != 0 and phase[1] == 0]
        elif want == "col":
            options = [phase for phase in phases if phase[0] == 0 and phase[1] != 0]
        else:
            options = [phase for phase in phases if phase[0] != 0 and phase[1] != 0]
        if options:
            modes.extend(options)
    return modes


def _sample_shift(tile: np.ndarray) -> tuple[int, int]:
    pr, pc = tile.shape
    phases = _shift_modes(pr, pc)
    random.shuffle(phases)
    for shift in phases:
        changed = _expand_tile(tile, tile.shape, shift) != tile
        changed_count = int(changed.sum())
        hidden_count = tile.size - changed_count
        if 0 < hidden_count < tile.size and changed_count >= max(2, tile.size // 4):
            return shift
    raise RuntimeError("failed to sample a suitable phase shift")


def _sample_rectangle(changed: np.ndarray) -> tuple[int, int, int, int]:
    H, W = changed.shape
    for _ in range(400):
        h = random.randint(3, min(H - 1, max(3, H // 2 + 2)))
        w = random.randint(3, min(W - 1, max(3, W // 2 + 2)))
        r0 = random.randint(0, H - h)
        c0 = random.randint(0, W - w)
        r1 = r0 + h - 1
        c1 = c0 + w - 1
        region = changed[r0 : r1 + 1, c0 : c1 + 1]
        visible = int(region.sum())
        area = h * w
        hidden = area - visible
        if hidden < 1 or visible < max(4, area // 4):
            continue
        if not region[0, :].any() or not region[-1, :].any():
            continue
        if not region[:, 0].any() or not region[:, -1].any():
            continue
        return r0, c0, r1, c1
    raise RuntimeError("failed to sample a visible-border rectangle")


def _is_far_from_noise(pos: tuple[int, int], used: set[tuple[int, int]]) -> bool:
    r, c = pos
    for rr, cc in used:
        if (r, c) == (rr, cc):
            return False
        for nr, nc in neighbors8(rr, cc):
            if (nr, nc) == (r, c):
                return False
    return True


def _sample_noise_cells(
    base: np.ndarray, tile: np.ndarray, rect: tuple[int, int, int, int], count: int
) -> list[tuple[int, int, int]]:
    H, W = base.shape
    r0, c0, r1, c1 = rect
    candidates = [(r, c) for r in range(H) for c in range(W) if not (r0 <= r <= r1 and c0 <= c <= c1)]
    random.shuffle(candidates)
    chosen: list[tuple[int, int, int]] = []
    used_pos: set[tuple[int, int]] = set()
    for r, c in candidates:
        if not _is_far_from_noise((r, c), used_pos):
            continue
        blocked = {int(base[r, c])}
        pr, pc = tile.shape
        for sr in range(pr):
            for sc in range(pc):
                blocked.add(int(tile[(r + sr) % pr, (c + sc) % pc]))
        allowed = [color for color in INPUT_COLORS if color not in blocked]
        if not allowed:
            continue
        color = random.choice(allowed)
        chosen.append((r, c, color))
        used_pos.add((r, c))
        if len(chosen) == count:
            return chosen
    raise RuntimeError("failed to place isolated noise")


def _expected_output(grid: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    r0, c0, r1, c1 = rect
    out[r0 : r1 + 1, c0 : c1 + 1] = OUTPUT_COLOR
    return out


def _generate_case(return_meta: bool = False):
    for _ in range(GEN_ATTEMPTS):
        pr = random.randint(2, 5)
        pc = random.randint(2, 5)
        H = random.randint(max(10, 3 * pr), 24)
        W = random.randint(max(10, 3 * pc), 24)
        tile = _sample_tile(pr, pc)
        try:
            shift = _sample_shift(tile)
        except RuntimeError:
            continue
        base = _expand_tile(tile, (H, W))
        shifted = _expand_tile(tile, (H, W), shift)
        changed = shifted != base
        rect = _sample_rectangle(changed)

        grid = base.copy()
        r0, c0, r1, c1 = rect
        grid[r0 : r1 + 1, c0 : c1 + 1] = shifted[r0 : r1 + 1, c0 : c1 + 1]

        noise_count = random.randint(1, 3)
        noise = _sample_noise_cells(base, tile, rect, noise_count)
        for r, c, color in noise:
            grid[r, c] = color

        expected = _expected_output(grid, rect)
        try:
            info = _solve_details(grid)
            actual = main(grid)
        except AssertionError:
            continue

        hidden_mask = base[r0 : r1 + 1, c0 : c1 + 1] == shifted[r0 : r1 + 1, c0 : c1 + 1]
        if int(hidden_mask.sum()) < 1:
            continue
        if info["rect"] != rect or info["shift"] != shift:
            continue
        if info["runner_up_score"] is not None and info["runner_up_score"] >= info["score"]:
            continue
        if not np.array_equal(actual, expected):
            continue
        if any(actual[r, c] != grid[r, c] for r, c, _color in noise):
            continue

        if return_meta:
            return grid, {
                "rect": rect,
                "shift": shift,
                "period": (pr, pc),
                "tile": tile,
                "noise": noise,
            }
        return grid
    raise RuntimeError("failed to generate a valid input")


def generate_input() -> np.ndarray:
    return _generate_case(return_meta=False)
