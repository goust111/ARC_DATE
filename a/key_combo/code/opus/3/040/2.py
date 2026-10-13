import random
from collections import Counter

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    """Return the most common color in the grid."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude=()) -> list[int]:
    """Return sorted unique colors, optionally excluding some values."""
    blocked = {int(x) for x in exclude}
    return sorted(int(x) for x in set(map(int, grid.ravel())) if int(x) not in blocked)


def _compress_panel(panel: np.ndarray, fill: int, marker: int) -> np.ndarray:
    h, w = panel.shape
    assert h % 3 == 0 and w % 3 == 0, "panel dimensions must be divisible by 3"
    block_h = h // 3
    block_w = w // 3
    out = np.full((3, 3), int(fill), dtype=int)
    for br in range(3):
        for bc in range(3):
            r0 = br * block_h
            c0 = bc * block_w
            block = panel[r0 : r0 + block_h, c0 : c0 + block_w]
            if np.any(block == marker):
                out[br, bc] = int(marker)
    return out


def _shift_mask(mask: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """Translate a boolean 3x3 mask with clipping at the border."""
    out = np.zeros_like(mask, dtype=bool)
    rows, cols = np.nonzero(mask)
    rows2 = rows + int(dr)
    cols2 = cols + int(dc)
    keep = (0 <= rows2) & (rows2 < mask.shape[0]) & (0 <= cols2) & (cols2 < mask.shape[1])
    out[rows2[keep], cols2[keep]] = True
    return out


def _detect_shift(frames: list[np.ndarray], fill: int, marker: int) -> tuple[int, int]:
    masks = [frame == marker for frame in frames]
    candidates = []
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            if np.array_equal(_shift_mask(masks[0], dr, dc), masks[1]) and np.array_equal(
                _shift_mask(masks[1], dr, dc), masks[2]
            ):
                candidates.append((dr, dc))
    assert len(candidates) == 1, f"expected a unique coarse translation, got {candidates}"
    return candidates[0]


def _mask_to_grid(mask: np.ndarray, fill: int, marker: int) -> np.ndarray:
    out = np.full(mask.shape, int(fill), dtype=int)
    out[mask] = int(marker)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Compress each panel to a 3x3 marker map, read the constant motion, and extrapolate one more frame.

    Input structure:
    - The input is a wide grid made of three equal-width panels placed side by side.
    - All panels use the same dominant fill color and one sparse marker color.
    - Each panel height and width are divisible by 3, so each panel can be partitioned into a 3x3 array of equal blocks.

    Rule:
    - Compress each panel independently: a coarse cell becomes the marker color iff its block contains at least one marker pixel; otherwise it becomes the fill color.
    - Treat the three compressed 3x3 grids as consecutive frames and find the unique translation (dr, dc) whose clipped shift maps frame 1 to frame 2 and frame 2 to frame 3.
    - Apply the same clipped translation once more to frame 3 and output that single 3x3 extrapolated frame.

    Key insight:
    - The raw marker pixels jitter inside each coarse block, so the motion pattern is only visible after panel-wise compression to the abstract 3x3 representation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    fill = get_bg(g)
    palette = colors_present(g)
    assert len(palette) == 2, "expected exactly two colors in the input"
    marker = next(color for color in palette if color != fill)

    h, w = g.shape
    assert w % 3 == 0, "input must contain exactly three equal-width panels"
    panel_w = w // 3
    assert h % 3 == 0 and panel_w % 3 == 0, "each panel must be divisible into a 3x3 grid"

    frames = [
        _compress_panel(g[:, idx * panel_w : (idx + 1) * panel_w], fill, marker)
        for idx in range(3)
    ]
    dr, dc = _detect_shift(frames, fill, marker)
    out_mask = _shift_mask(frames[2] == marker, dr, dc)
    return _mask_to_grid(out_mask, fill, marker)


_PANEL_HEIGHTS = (6, 9, 12)
_PANEL_WIDTHS = (6, 9)
_SHIFT_OPTIONS = [
    (0, 0),
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
]
_GEN_ATTEMPTS = 400


def _sample_shift() -> tuple[int, int]:
    if random.random() < 0.55:
        return 0, 0
    return random.choice(_SHIFT_OPTIONS[1:])


def _sample_start_mask() -> np.ndarray:
    mask = np.zeros((3, 3), dtype=bool)
    count = random.randint(1, 6)
    cells = random.sample(range(9), count)
    for cell in cells:
        mask[cell // 3, cell % 3] = True
    return mask


def _sample_frames() -> tuple[list[np.ndarray], tuple[int, int]]:
    for _ in range(200):
        dr, dc = _sample_shift()
        frame1 = _sample_start_mask()
        frame2 = _shift_mask(frame1, dr, dc)
        frame3 = _shift_mask(frame2, dr, dc)
        if not frame1.any() or not frame2.any() or not frame3.any():
            continue
        if (dr, dc) != (0, 0) and np.array_equal(frame1, frame2) and np.array_equal(frame2, frame3):
            continue
        try:
            _detect_shift(
                [_mask_to_grid(frame1, 0, 1), _mask_to_grid(frame2, 0, 1), _mask_to_grid(frame3, 0, 1)],
                0,
                1,
            )
        except AssertionError:
            continue
        return [frame1, frame2, frame3], (dr, dc)
    raise RuntimeError("failed to sample a valid compressed frame sequence")


def _sample_local_pattern(block_h: int, block_w: int, used: set[tuple[int, ...]]) -> tuple[int, ...]:
    area = block_h * block_w
    max_points = min(area, 2 if area <= 4 else 4)
    for _ in range(60):
        size = random.randint(1, max_points)
        picked = tuple(sorted(random.sample(range(area), size)))
        if picked not in used:
            return picked
    return tuple(sorted(random.sample(range(area), random.randint(1, max_points))))


def _render_panels(
    frames: list[np.ndarray], panel_h: int, panel_w: int, fill: int, marker: int
) -> list[np.ndarray]:
    block_h = panel_h // 3
    block_w = panel_w // 3
    panels = [np.full((panel_h, panel_w), int(fill), dtype=int) for _ in range(3)]

    for br in range(3):
        for bc in range(3):
            active = [idx for idx, frame in enumerate(frames) if frame[br, bc]]
            if not active:
                continue
            used_patterns: set[tuple[int, ...]] = set()
            r0 = br * block_h
            c0 = bc * block_w
            for idx in active:
                picked = _sample_local_pattern(block_h, block_w, used_patterns)
                used_patterns.add(picked)
                for cell in picked:
                    rr = r0 + cell // block_w
                    cc = c0 + cell % block_w
                    panels[idx][rr, cc] = int(marker)

    return panels


def _valid_rendered_input(grid: np.ndarray, fill: int, marker: int, frames: list[np.ndarray]) -> bool:
    h, w = grid.shape
    panel_w = w // 3
    if get_bg(grid) != fill:
        return False
    if colors_present(grid) != sorted([fill, marker]):
        return False
    recompressed = [
        _compress_panel(grid[:, idx * panel_w : (idx + 1) * panel_w], fill, marker) == marker
        for idx in range(3)
    ]
    if not all(np.array_equal(a, b) for a, b in zip(recompressed, frames)):
        return False
    panels = [grid[:, idx * panel_w : (idx + 1) * panel_w] for idx in range(3)]
    if any(np.array_equal(panels[i], panels[j]) for i in range(3) for j in range(i + 1, 3)):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        panel_h = random.choice(_PANEL_HEIGHTS)
        panel_w = random.choice(_PANEL_WIDTHS)
        fill, marker = random.sample(range(10), 2)

        frames, _shift = _sample_frames()
        panels = _render_panels(frames, panel_h, panel_w, fill, marker)
        grid = np.concatenate(panels, axis=1)

        if not _valid_rendered_input(grid, fill, marker, frames):
            continue
        out = main(grid)
        if out.shape != (3, 3):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input for puzzle 2")
