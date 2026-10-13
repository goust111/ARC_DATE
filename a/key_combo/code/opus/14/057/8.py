from __future__ import annotations

import numpy as np


GRID_SHAPE = (16, 16)
BAND_HEIGHT = 4
BAND_COLORS = (2, 1, 3, 4)
PRIORITY = (0, 1, 3, 2)
NATURAL_PRIORITY = (0, 1, 2, 3)
LOCAL_ROWS = (1, 2)
SLOT_COLS = (2, 6, 10, 14)
SHAPES = (
    ((-1, 0), (0, -1), (0, 1), (1, 0)),
)
STATE_COMPLETE = "complete"
STATE_CENTER_ONLY = "center_only"
STATE_SHELL_ONLY = "shell_only"
def _band_slice(index: int) -> slice:
    start = index * BAND_HEIGHT
    return slice(start, start + BAND_HEIGHT)


def _shape_signature(offsets: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(dr), int(dc)) for dr, dc in offsets))


def _candidate_offsets(band: np.ndarray, color: int) -> set[tuple[tuple[int, int], ...]]:
    candidates: set[tuple[tuple[int, int], ...]] = set()
    for r in range(1, band.shape[0] - 1):
        for c in range(1, band.shape[1] - 1):
            if int(band[r, c]) != color:
                continue
            offsets = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    if int(band[r + dr, c + dc]) == color:
                        offsets.append((dr, dc))
            offset_set = frozenset(offsets)
            if len(offset_set) < 3:
                continue
            if any((-dr, -dc) not in offset_set for dr, dc in offset_set):
                continue
            candidates.add(_shape_signature(tuple(offset_set)))
    return candidates


def _infer_shell_offsets(grid: np.ndarray) -> tuple[tuple[int, int], ...]:
    candidates: set[tuple[tuple[int, int], ...]] = set()
    for band_index, color in enumerate(BAND_COLORS):
        band = grid[_band_slice(band_index), :]
        candidates |= _candidate_offsets(band, color)
    assert len(candidates) == 1, f"expected one global shell pattern, found {sorted(candidates)}"
    return next(iter(candidates))


def _classify_centers(
    band: np.ndarray, color: int, offsets: tuple[tuple[int, int], ...]
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    center_only: list[tuple[int, int]] = []
    shell_only: list[tuple[int, int]] = []
    target = _shape_signature(offsets)
    for r in range(1, band.shape[0] - 1):
        for c in range(1, band.shape[1] - 1):
            local_offsets = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    if int(band[r + dr, c + dc]) == color:
                        local_offsets.append((dr, dc))
            signature = _shape_signature(tuple(local_offsets))
            has_center = int(band[r, c]) == color
            if has_center and signature == ():
                center_only.append((r, c))
            elif not has_center and signature == target:
                shell_only.append((r, c))
    return center_only, shell_only


def _complete_band(
    band: np.ndarray, color: int, offsets: tuple[tuple[int, int], ...]
) -> np.ndarray:
    out = band.copy()
    center_only, shell_only = _classify_centers(band, color, offsets)
    for r, c in center_only:
        for dr, dc in offsets:
            rr, cc = r + dr, c + dc
            if int(out[rr, cc]) == 0:
                out[rr, cc] = color
    for r, c in shell_only:
        if int(out[r, c]) == 0:
            out[r, c] = color
    return out


def _overlay_bands(bands: list[np.ndarray], priority: tuple[int, int, int, int]) -> np.ndarray:
    out = np.zeros_like(bands[0])
    for band_index in reversed(priority):
        mask = bands[band_index] != 0
        out[mask] = bands[band_index][mask]
    return out


def _validate_input_grid(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.shape == GRID_SHAPE, f"expected 16x16 grid, got {grid.shape}"
    for band_index, color in enumerate(BAND_COLORS):
        band = grid[_band_slice(band_index), :]
        assert np.all((band == 0) | (band == color)), "each band must be monochrome on black"


def main(grid: np.ndarray) -> np.ndarray:
    """Complete motifs inside each horizontal band, then collapse the four completed bands with a swapped lower-layer priority.

    Input structure:
    - The input is a 16x16 black grid split into four 4x16 horizontal bands.
    - Band colors are fixed by band: top red(2), second blue(1), third green(3), bottom yellow(4).
    - Each band contains several separated 3x3 same-color motifs that all share one global shell-offset pattern; some copies are complete, some are center-only, and some are shell-only.

    Rule:
    - Learn the single shell-offset pattern from any complete motif visible anywhere in the four bands.
    - In every band, complete each center-only copy by adding its missing shell pixels on black cells, and complete each shell-only copy by restoring the missing center pixel.
    - Overlay the four completed bands at matching local coordinates to produce one 4x16 output band, using priority band1 > band2 > band4 > band3.

    Key insight:
    - The template is shared across bands, so completion must happen before compositing; only then does the non-natural band order resolve cross-band conflicts correctly.
    """
    g = np.asarray(grid, dtype=int)
    _validate_input_grid(g)
    offsets = _infer_shell_offsets(g)
    completed = [
        _complete_band(g[_band_slice(i), :], BAND_COLORS[i], offsets)
        for i in range(4)
    ]
    return _overlay_bands(completed, PRIORITY)


def _paint_state(
    band: np.ndarray,
    color: int,
    center: tuple[int, int],
    offsets: tuple[tuple[int, int], ...],
    state: str,
) -> None:
    r, c = center
    if state in (STATE_COMPLETE, STATE_CENTER_ONLY):
        band[r, c] = color
    if state in (STATE_COMPLETE, STATE_SHELL_ONLY):
        for dr, dc in offsets:
            band[r + dr, c + dc] = color


def _build_case() -> tuple[np.ndarray, tuple[tuple[int, int], ...], dict[str, tuple[int, int]], str]:
    offsets = _shape_signature(SHAPES[int(np.random.randint(len(SHAPES)))])
    rows = [int(np.random.choice(LOCAL_ROWS)) for _ in SLOT_COLS]
    slot_centers = [(rows[i], SLOT_COLS[i]) for i in range(len(SLOT_COLS))]
    order = [int(i) for i in np.random.permutation(len(SLOT_COLS))]
    role_to_slot = {
        "p": order[0],
        "q": order[1],
        "r": order[2],
        "s": order[3],
    }
    role_centers = {role: slot_centers[slot_index] for role, slot_index in role_to_slot.items()}
    band3_hidden_partial = STATE_CENTER_ONLY if np.random.rand() < 0.5 else STATE_SHELL_ONLY

    bands = [np.zeros((BAND_HEIGHT, GRID_SHAPE[1]), dtype=int) for _ in range(4)]

    assignments = {
        0: {"r": STATE_COMPLETE, "s": STATE_SHELL_ONLY},
        1: {"q": STATE_SHELL_ONLY, "r": STATE_COMPLETE},
        2: {"p": STATE_COMPLETE, "q": STATE_COMPLETE, "r": band3_hidden_partial, "s": STATE_COMPLETE},
        3: {"p": STATE_CENTER_ONLY, "r": STATE_COMPLETE},
    }
    for band_index, role_map in assignments.items():
        color = BAND_COLORS[band_index]
        for role, state in role_map.items():
            _paint_state(
                bands[band_index],
                color,
                role_centers[role],
                offsets,
                state,
            )

    grid = np.zeros(GRID_SHAPE, dtype=int)
    for band_index, band in enumerate(bands):
        grid[_band_slice(band_index), :] = band
    return grid, offsets, role_centers, band3_hidden_partial


def _raw_bands(grid: np.ndarray) -> list[np.ndarray]:
    return [grid[_band_slice(i), :].copy() for i in range(4)]


def _complete_subset(
    bands: list[np.ndarray],
    offsets: tuple[tuple[int, int], ...],
    *,
    fill_center_only: bool,
    fill_shell_only: bool,
) -> list[np.ndarray]:
    out = [band.copy() for band in bands]
    for i, color in enumerate(BAND_COLORS):
        center_only, shell_only = _classify_centers(bands[i], color, offsets)
        if fill_center_only:
            for r, c in center_only:
                for dr, dc in offsets:
                    rr, cc = r + dr, c + dc
                    if int(out[i][rr, cc]) == 0:
                        out[i][rr, cc] = color
        if fill_shell_only:
            for r, c in shell_only:
                if int(out[i][r, c]) == 0:
                    out[i][r, c] = color
    return out


def _role_patch(center: tuple[int, int], offsets: tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
    r, c = center
    return {(r, c)} | {(r + dr, c + dc) for dr, dc in offsets}


def _assert_semantics(
    grid: np.ndarray,
    offsets: tuple[tuple[int, int], ...],
    role_centers: dict[str, tuple[int, int]],
) -> None:
    _validate_input_grid(grid)
    inferred = _infer_shell_offsets(grid)
    assert inferred == offsets

    bands = _raw_bands(grid)
    completed = [_complete_band(bands[i], BAND_COLORS[i], offsets) for i in range(4)]
    expected = _overlay_bands(completed, PRIORITY)
    assert np.array_equal(main(grid), expected)

    raw_overlay = _overlay_bands(bands, PRIORITY)
    only_center_overlay = _overlay_bands(
        _complete_subset(bands, offsets, fill_center_only=True, fill_shell_only=False),
        PRIORITY,
    )
    only_shell_overlay = _overlay_bands(
        _complete_subset(bands, offsets, fill_center_only=False, fill_shell_only=True),
        PRIORITY,
    )
    natural_overlay = _overlay_bands(completed, NATURAL_PRIORITY)

    assert not np.array_equal(raw_overlay, expected), "skipping all completion must change the result"
    assert not np.array_equal(only_center_overlay, expected), "shell-only completion must matter"
    assert not np.array_equal(only_shell_overlay, expected), "center-only completion must matter"
    assert not np.array_equal(natural_overlay, expected), "swapping bands 3 and 4 must matter"

    p_center = role_centers["p"]
    q_center = role_centers["q"]
    s_center = role_centers["s"]

    p_patch = _role_patch(p_center, offsets)
    q_patch = _role_patch(q_center, offsets)
    s_patch = _role_patch(s_center, offsets)

    for r, c in p_patch:
        assert int(expected[r, c]) == 4, "band4 must beat band3 on the dedicated conflict patch"
    assert int(raw_overlay[p_center]) == 4 and any(int(raw_overlay[r, c]) == 3 for r, c in p_patch - {p_center})
    assert int(expected[q_center]) == 1, "shell-only band2 center must override band3 at Q"
    assert int(raw_overlay[q_center]) == 3, "before shell-only completion, Q must expose the lower green center"
    assert all(int(expected[r, c]) == 1 for r, c in q_patch), "after completion, Q should become a full blue motif"
    assert int(expected[s_center]) == 2, "shell-only band1 center must override band3 at S"
    assert int(raw_overlay[s_center]) == 3, "before shell-only completion, S must expose the lower green center"
    assert all(int(expected[r, c]) == 2 for r, c in s_patch), "after completion, S should become a full red motif"

    for band_index, color in enumerate(BAND_COLORS):
        center_only, shell_only = _classify_centers(bands[band_index], color, offsets)
        complete_count = 0
        for r in range(1, BAND_HEIGHT - 1):
            for c in range(1, grid.shape[1] - 1):
                if int(bands[band_index][r, c]) != color:
                    continue
                if all(int(bands[band_index][r + dr, c + dc]) == color for dr, dc in offsets):
                    complete_count += 1
        assert complete_count >= 1
        assert len(center_only) + len(shell_only) >= 1


def _gen_attempt() -> np.ndarray | None:
    grid, offsets, role_centers, _hidden_kind = _build_case()
    try:
        _assert_semantics(grid, offsets, role_centers)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(200):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
