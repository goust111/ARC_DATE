from __future__ import annotations

import numpy as np


def _frame_from_color(grid: np.ndarray, color: int) -> dict:
    coords = np.argwhere(grid == int(color))
    assert coords.size > 0, "expected each foreground color to be present"

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert h >= 3 and w >= 3, "frames must have an interior"

    rel = coords - np.array([r0, c0])
    on_border = (
        (rel[:, 0] == 0)
        | (rel[:, 0] == h - 1)
        | (rel[:, 1] == 0)
        | (rel[:, 1] == w - 1)
    )
    expected_pixels = 2 * h + 2 * w - 4
    assert int(coords.shape[0]) == expected_pixels, "foreground color must form a full perimeter"
    assert bool(np.all(on_border)), "foreground color must stay on its bounding-box border"

    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "height": h,
        "width": w,
        "area": h * w,
    }


def _extract_frames(grid: np.ndarray) -> list[dict]:
    frames = [_frame_from_color(grid, int(color)) for color in np.unique(grid) if int(color) != 0]
    assert 2 <= len(frames) <= 4, "expected a 2-4 layer nesting chain"

    frames.sort(key=lambda item: int(item["area"]), reverse=True)
    for outer, inner in zip(frames, frames[1:]):
        or0, oc0, or1, oc1 = outer["bbox"]
        ir0, ic0, ir1, ic1 = inner["bbox"]
        assert or0 < ir0 < ir1 < or1, "frames must be strictly nested vertically"
        assert oc0 < ic0 < ic1 < oc1, "frames must be strictly nested horizontally"

    return frames


def _draw_frame(out: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    out[top, left : left + width] = int(color)
    out[top + height - 1, left : left + width] = int(color)
    out[top : top + height, left] = int(color)
    out[top : top + height, left + width - 1] = int(color)


def _render_flat_frames(frames: list[dict]) -> np.ndarray:
    max_height = max(int(frame["height"]) for frame in frames)
    total_width = sum(int(frame["width"]) for frame in frames) + len(frames) - 1
    out = np.zeros((max_height, total_width), dtype=int)

    col = 0
    for frame in frames:
        _draw_frame(
            out,
            0,
            col,
            int(frame["height"]),
            int(frame["width"]),
            int(frame["color"]),
        )
        col += int(frame["width"]) + 1

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Flatten a nesting chain of rectangular frames into a top-aligned row."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    frames = _extract_frames(g)
    return _render_flat_frames(frames)


_RNG = np.random.default_rng()

_LAYER_OPTIONS = (2, 3, 4)
_SHAPE_MODES = ("balanced", "wide", "tall")
_INNER_MIN = 3
_INNER_MAX = 10
_INSET_RANGE = (2, 4)
_MARGIN_RANGE = (0, 4)
_MAX_OUTER_HEIGHT = 22
_MAX_OUTER_WIDTH = 22
_GENERATION_ATTEMPTS = 300
_COLOR_POOL = np.arange(1, 10)


def _randint(bounds: tuple[int, int]) -> int:
    lo, hi = bounds
    return int(_RNG.integers(int(lo), int(hi) + 1))


def _shape_candidates(mode: str) -> list[tuple[int, int]]:
    candidates = []
    for h in range(_INNER_MIN, _INNER_MAX + 1):
        for w in range(_INNER_MIN, _INNER_MAX + 1):
            if mode == "balanced" and abs(h - w) <= 2:
                candidates.append((h, w))
            elif mode == "wide" and w >= h + 3:
                candidates.append((h, w))
            elif mode == "tall" and h >= w + 3:
                candidates.append((h, w))
    return candidates


def _frame_specs_from_samples(
    colors: np.ndarray,
    inner_h: int,
    inner_w: int,
    insets_r: list[int],
    insets_c: list[int],
) -> list[dict]:
    outer_h = int(inner_h + 2 * sum(insets_r))
    outer_w = int(inner_w + 2 * sum(insets_c))

    frames = []
    current_h = outer_h
    current_w = outer_w
    for i, color in enumerate(colors):
        frames.append(
            {
                "color": int(color),
                "height": int(current_h),
                "width": int(current_w),
                "area": int(current_h * current_w),
            }
        )
        if i + 1 < len(colors):
            current_h -= 2 * int(insets_r[i])
            current_w -= 2 * int(insets_c[i])

    assert current_h == inner_h and current_w == inner_w
    return frames


def _build_input(
    frames: list[dict],
    insets_r: list[int],
    insets_c: list[int],
    top_margin: int,
    left_margin: int,
    bottom_margin: int,
    right_margin: int,
) -> np.ndarray:
    outer_h = int(frames[0]["height"])
    outer_w = int(frames[0]["width"])
    grid = np.zeros(
        (top_margin + outer_h + bottom_margin, left_margin + outer_w + right_margin),
        dtype=int,
    )

    top = int(top_margin)
    left = int(left_margin)
    for i, frame in enumerate(frames):
        _draw_frame(
            grid,
            top,
            left,
            int(frame["height"]),
            int(frame["width"]),
            int(frame["color"]),
        )
        if i + 1 < len(frames):
            top += int(insets_r[i])
            left += int(insets_c[i])

    return grid


def _gen_attempt() -> np.ndarray | None:
    n_layers = int(_RNG.choice(_LAYER_OPTIONS))
    mode = str(_RNG.choice(_SHAPE_MODES))
    candidates = _shape_candidates(mode)
    if not candidates:
        return None

    inner_h, inner_w = candidates[int(_RNG.integers(len(candidates)))]
    insets_r = [_randint(_INSET_RANGE) for _ in range(n_layers - 1)]
    insets_c = [_randint(_INSET_RANGE) for _ in range(n_layers - 1)]
    outer_h = int(inner_h + 2 * sum(insets_r))
    outer_w = int(inner_w + 2 * sum(insets_c))
    if outer_h > _MAX_OUTER_HEIGHT or outer_w > _MAX_OUTER_WIDTH:
        return None

    colors = _RNG.choice(_COLOR_POOL, size=n_layers, replace=False)
    frames = _frame_specs_from_samples(colors, inner_h, inner_w, insets_r, insets_c)
    expected = _render_flat_frames(frames)

    margins = [_randint(_MARGIN_RANGE) for _ in range(4)]
    inp = _build_input(frames, insets_r, insets_c, *margins)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
