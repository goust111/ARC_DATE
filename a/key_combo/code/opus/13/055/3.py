from __future__ import annotations

import hashlib
import itertools
import random

import numpy as np


_BG = 0
_BLUE = 1
_RED = 2
_GREEN = 3
_YELLOW = 4
_GREY = 5
_ORANGE = 7

_Q = 5
_INPUT_SIZE = 2 * _Q + 1
_QUADRANT_COLORS = (_RED, _BLUE, _GREEN, _ORANGE)
_PAIR_OPTIONS = tuple((i, j) for i in range(4) for j in range(i + 1, 4))
_GEN_SEEN_INPUT_HASHES: set[str] = set()
_GEN_SEEN_OUTPUT_HASHES: set[str] = set()


def _extract_quadrants(grid: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, grid.shape)
    assert h == w and h % 2 == 1, "expected an odd square grid"
    mid = h // 2
    assert np.all(grid[mid, :] == _GREY), "expected a solid grey separator row"
    assert np.all(grid[:, mid] == _GREY), "expected a solid grey separator column"
    q = mid
    assert q == h - mid - 1, "separator must split the grid into equal quadrants"
    return [
        grid[:q, :q].copy(),
        grid[:q, mid + 1 :].copy(),
        grid[mid + 1 :, :q].copy(),
        grid[mid + 1 :, mid + 1 :].copy(),
    ]


def _overlay_quadrants(quadrants: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    q = int(quadrants[0].shape[0])
    merged = np.full((q, q), _BG, dtype=int)
    owner = np.full((q, q), -1, dtype=int)
    for idx, region in enumerate(quadrants):
        mask = (owner < 0) & (region != _BG)
        merged[mask] = region[mask]
        owner[mask] = idx
    return merged, owner


def _diagonal_runs(mask: np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    occ = np.asarray(mask, dtype=bool)
    h, w = map(int, occ.shape)
    runs: list[tuple[tuple[int, int], ...]] = []
    for r, c in map(tuple, np.argwhere(occ)):
        r = int(r)
        c = int(c)
        if r > 0 and c > 0 and occ[r - 1, c - 1]:
            continue
        run: list[tuple[int, int]] = []
        rr, cc = r, c
        while rr < h and cc < w and occ[rr, cc]:
            run.append((int(rr), int(cc)))
            rr += 1
            cc += 1
        runs.append(tuple(run))
    return sorted(runs)


def _recolor_diagonal_runs(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for run in _diagonal_runs(out != _BG):
        for idx, (r, c) in enumerate(run):
            if idx % 2 == 1:
                out[r, c] = _YELLOW
    return out


def _main_diagonal_pattern(mask: np.ndarray) -> str:
    return "".join("1" if bool(mask[i, i]) else "0" for i in range(_Q))


def _has_orthogonal_contact(mask: np.ndarray) -> bool:
    occ = set(map(tuple, np.argwhere(mask)))
    for r, c in occ:
        if (r + 1, c) in occ or (r, c + 1) in occ:
            return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the four colored quadrants and recolor every odd step of each emergent down-right diagonal run.

    Input structure:
    - The input is an odd square grid split by one full grey row and one full grey column through the center.
    - The cross separates four equal quadrants, each with black background and exactly one quadrant-specific foreground color.
    - Within each quadrant, the foreground pixels form short down-right diagonal segments.

    Rule:
    - Remove the grey separator cross and align the four quadrants onto one common 5x5 canvas.
    - Merge corresponding cells in priority order top-left, top-right, bottom-left, bottom-right, taking the first non-black pixel.
    - In the merged grid, find every maximal down-right non-black diagonal run; within each run, keep even-indexed pixels unchanged and recolor odd-indexed pixels yellow.

    Key insight:
    - The decisive diagonal lengths are computed only after the quadrant overlay, so lower-priority fragments can extend a higher-level shape and shift the odd/even recoloring boundary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    quadrants = _extract_quadrants(g)
    for region, color in zip(quadrants, _QUADRANT_COLORS):
        colors = set(map(int, np.unique(region)))
        assert colors <= {_BG, color}, "each quadrant must use only black plus its own foreground color"
        assert color in colors, "each quadrant must contain foreground pixels"

    merged, _owner = _overlay_quadrants(quadrants)
    return _recolor_diagonal_runs(merged)


def _build_quadrant_catalog() -> dict[str, list[dict[str, object]]]:
    segments: list[tuple[tuple[int, int], ...]] = []
    for r in range(_Q):
        for c in range(_Q):
            for length in range(2, 5):
                if r + length <= _Q and c + length <= _Q:
                    segments.append(tuple((r + i, c + i) for i in range(length)))
    segments = list(dict.fromkeys(segments))

    catalog: dict[str, list[dict[str, object]]] = {}
    seen_masks: set[bytes] = set()
    for count in (1, 2, 3):
        for combo in itertools.combinations(range(len(segments)), count):
            mask = np.zeros((_Q, _Q), dtype=np.uint8)
            valid = True
            for idx in combo:
                for r, c in segments[idx]:
                    if mask[r, c]:
                        valid = False
                        break
                    mask[r, c] = 1
                if not valid:
                    break
            if not valid or _has_orthogonal_contact(mask):
                continue

            runs = _diagonal_runs(mask)
            if not (1 <= len(runs) <= 3 and all(2 <= len(run) <= 4 for run in runs)):
                continue

            key = bytes(mask.reshape(-1))
            if key in seen_masks:
                continue
            seen_masks.add(key)

            pattern = _main_diagonal_pattern(mask)
            catalog.setdefault(pattern, []).append(
                {
                    "mask": mask.copy(),
                    "run_count": len(runs),
                    "runs": runs,
                    "area": int(mask.sum()),
                }
            )
    return catalog


_CATALOG_BY_PATTERN = _build_quadrant_catalog()


def _choose_catalog_entry(pattern: str) -> dict[str, object]:
    entries = _CATALOG_BY_PATTERN[pattern]
    desired_run_count = random.choice((1, 2, 3))
    matching = [entry for entry in entries if int(entry["run_count"]) == desired_run_count]
    return random.choice(matching if matching else entries)


def _assemble_input(quadrants: list[np.ndarray]) -> np.ndarray:
    grid = np.full((_INPUT_SIZE, _INPUT_SIZE), _BG, dtype=int)
    grid[_Q, :] = _GREY
    grid[:, _Q] = _GREY
    grid[:_Q, :_Q] = quadrants[0]
    grid[:_Q, _Q + 1 :] = quadrants[1]
    grid[_Q + 1 :, :_Q] = quadrants[2]
    grid[_Q + 1 :, _Q + 1 :] = quadrants[3]
    return grid


def _standalone_recolor(mask: np.ndarray, color: int) -> np.ndarray:
    return _recolor_diagonal_runs(np.where(mask, int(color), _BG))


def _validate_example(grid: np.ndarray) -> bool:
    try:
        g = np.asarray(grid, dtype=int)
        if g.shape != (_INPUT_SIZE, _INPUT_SIZE):
            return False

        quadrants = _extract_quadrants(g)
        masks: list[np.ndarray] = []
        max_source_run = 0
        for region, color in zip(quadrants, _QUADRANT_COLORS):
            colors = set(map(int, np.unique(region)))
            if colors - {_BG, color} or color not in colors:
                return False
            mask = region == color
            if _has_orthogonal_contact(mask):
                return False
            runs = _diagonal_runs(mask)
            if not (1 <= len(runs) <= 3 and all(2 <= len(run) <= 4 for run in runs)):
                return False
            max_source_run = max(max_source_run, max(len(run) for run in runs))
            masks.append(mask)

        merged, owner = _overlay_quadrants(quadrants)
        out = _recolor_diagonal_runs(merged)
        if not np.array_equal(main(g), out):
            return False

        visible_owners = set(map(int, owner[owner >= 0]))
        if visible_owners != {0, 1, 2, 3}:
            return False

        diag_owners = tuple(int(owner[i, i]) for i in range(_Q))
        if -1 in diag_owners:
            return False
        lead, trail = diag_owners[0], diag_owners[-1]
        if not (
            diag_owners[:3] == (lead, lead, lead)
            and diag_owners[3:] == (trail, trail)
            and lead < trail
            and lead != trail
        ):
            return False

        if _main_diagonal_pattern(masks[lead]) != "11100":
            return False
        if _main_diagonal_pattern(masks[trail]) != "01111":
            return False
        for idx, mask in enumerate(masks):
            if idx not in {lead, trail} and _main_diagonal_pattern(mask) != "00000":
                return False

        merged_runs = _diagonal_runs(merged != _BG)
        if len(merged_runs) < 2:
            return False
        if tuple((i, i) for i in range(_Q)) not in merged_runs:
            return False
        if max(len(run) for run in merged_runs) != _Q:
            return False
        if not max_source_run < _Q:
            return False

        visible_count = int(np.count_nonzero(merged))
        if not 7 <= visible_count <= 15:
            return False

        trail_local = _standalone_recolor(masks[trail], _QUADRANT_COLORS[trail])
        trail_visible_cells = ((3, 3), (4, 4))
        if not any(out[r, c] != trail_local[r, c] for r, c in trail_visible_cells):
            return False

        out_hash = hashlib.md5(out.tobytes() + str(out.shape).encode()).hexdigest()
        if out_hash in _GEN_SEEN_OUTPUT_HASHES:
            return False
        return True
    except Exception:
        return False


def _attempt_generate() -> np.ndarray | None:
    lead, trail = random.choice(_PAIR_OPTIONS)
    chosen: list[dict[str, object] | None] = [None, None, None, None]
    chosen[lead] = _choose_catalog_entry("11100")
    chosen[trail] = _choose_catalog_entry("01111")
    for idx in range(4):
        if chosen[idx] is None:
            chosen[idx] = _choose_catalog_entry("00000")

    quadrants = [
        np.where(np.asarray(chosen[idx]["mask"], dtype=bool), _QUADRANT_COLORS[idx], _BG)
        for idx in range(4)
    ]
    grid = _assemble_input(quadrants)
    if not _validate_example(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _attempt_generate()
        if grid is None:
            continue

        inp_hash = hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()
        out = main(grid)
        out_hash = hashlib.md5(out.tobytes() + str(out.shape).encode()).hexdigest()
        if inp_hash in _GEN_SEEN_INPUT_HASHES or out_hash in _GEN_SEEN_OUTPUT_HASHES:
            continue

        _GEN_SEEN_INPUT_HASHES.add(inp_hash)
        _GEN_SEEN_OUTPUT_HASHES.add(out_hash)
        return grid
    raise RuntimeError("failed to generate a valid non-degenerate input")
