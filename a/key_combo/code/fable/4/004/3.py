from __future__ import annotations

import numpy as np


_RNG = np.random.default_rng()
_ARC_COLORS = tuple(range(10))
_VIEW_NAMES = ("upper-left", "upper-right", "lower-left", "lower-right")
_GEN_RETRY_LIMIT = 600


def _split_quadrants(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= 2 and width >= 2, "input must contain four nonempty quadrants"

    if height % 2 == 1 and width % 2 == 1:
        mid_r = height // 2
        mid_c = width // 2
        sep = int(g[mid_r, mid_c])
        if np.all(g[mid_r, :] == sep) and np.all(g[:, mid_c] == sep):
            assert mid_r > 0 and mid_c > 0, "separator must leave nonempty quadrants"
            return (
                g[:mid_r, :mid_c],
                g[:mid_r, mid_c + 1 :],
                g[mid_r + 1 :, :mid_c],
                g[mid_r + 1 :, mid_c + 1 :],
            )

    assert height % 2 == 0 and width % 2 == 0, "input must split evenly or have one separator cross"
    half_h = height // 2
    half_w = width // 2
    return (
        g[:half_h, :half_w],
        g[:half_h, half_w:],
        g[half_h:, :half_w],
        g[half_h:, half_w:],
    )


def _align_views(views: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, ...]:
    ul, ur, ll, lr = views
    shape = ul.shape
    assert all(view.shape == shape for view in views), "quadrants must be equal size"
    return (ul, np.fliplr(ur), np.flipud(ll), np.flipud(np.fliplr(lr)))


def _candidate_output(stack: np.ndarray, hole_color: int) -> np.ndarray | None:
    count, height, width = map(int, stack.shape)
    assert count == 4
    out = np.empty((height, width), dtype=int)

    for r in range(height):
        for c in range(width):
            values = [int(v) for v in stack[:, r, c] if int(v) != int(hole_color)]
            if not values:
                return None
            first = values[0]
            if any(v != first for v in values[1:]):
                return None
            out[r, c] = first
    return out


def _solve_stack(stack: np.ndarray) -> tuple[int, np.ndarray]:
    candidates: list[tuple[int, np.ndarray]] = []
    for color in map(int, np.unique(stack)):
        if not all(np.any(stack[i] == color) for i in range(stack.shape[0])):
            continue
        out = _candidate_output(stack, color)
        if out is not None and not np.any(out == color):
            candidates.append((color, out))

    assert len(candidates) == 1, "expected a unique inferred occluding color"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align four mirrored stained views and fuse them into the hidden image.

    Input structure:
    - The input is a 2x2 arrangement of four equal-size views, optionally separated
      by one solid cross row and column.
    - The upper-left view is the hidden image P, the upper-right is P mirrored
      left-to-right, the lower-left is P mirrored top-to-bottom, and the
      lower-right is P rotated 180 degrees.
    - In every view, some pixels are replaced by one shared occluding color that
      is not identified in advance.

    Rule:
    - Try each visible color as the transparent occluding color.
    - Flip the upper-right view left-to-right, flip the lower-left view
      top-to-bottom, and rotate the lower-right view 180 degrees to align all
      views with the upper-left coordinates.
    - The real occluding color is the unique color for which every aligned cell
      has one non-occluding value and all non-occluding values agree; output
      those agreed values.

    Key insight:
    - The four quadrants are mirror views of one image, not four independent
      layers.  Only after applying the quadrant-specific flips does one color
      become a consistent transparent stain across all views.
    """
    views = _split_quadrants(grid)
    aligned = _align_views(views)
    stack = np.stack(aligned, axis=0)
    _hole_color, out = _solve_stack(stack)
    return out.astype(int)


def _make_hidden_pattern(height: int, width: int, colors: list[int]) -> np.ndarray:
    base = int(colors[0])
    detail_colors = [int(c) for c in colors[1:]]
    pattern = np.full((height, width), base, dtype=int)

    for color in detail_colors:
        rect_count = int(_RNG.integers(1, 3))
        for _ in range(rect_count):
            rh = int(_RNG.integers(1, max(2, min(4, height // 2 + 1)) + 1))
            rw = int(_RNG.integers(1, max(2, min(4, width // 2 + 1)) + 1))
            r0 = int(_RNG.integers(0, height - rh + 1))
            c0 = int(_RNG.integers(0, width - rw + 1))
            pattern[r0 : r0 + rh, c0 : c0 + rw] = color

    for color in detail_colors:
        if not np.any(pattern == color):
            r = int(_RNG.integers(0, height))
            c = int(_RNG.integers(0, width))
            pattern[r, c] = color

    # A few single cells break accidental mirror symmetries without making the image noisy.
    accent_count = int(_RNG.integers(2, 6))
    for _ in range(accent_count):
        r = int(_RNG.integers(0, height))
        c = int(_RNG.integers(0, width))
        pattern[r, c] = int(_RNG.choice(colors))

    return pattern


def _clear_one_view_if_all_masked(masks: np.ndarray, r: int, c: int, *, keep_masked: int | None = None) -> None:
    if not np.all(masks[:, r, c]):
        return
    choices = [i for i in range(4) if i != keep_masked]
    masks[int(_RNG.choice(choices)), r, c] = False


def _make_masks(pattern: np.ndarray) -> np.ndarray:
    height, width = map(int, pattern.shape)
    rate = float(_RNG.uniform(0.16, 0.31))
    masks = _RNG.random((4, height, width)) < rate

    for r, c in np.argwhere(np.all(masks, axis=0)):
        masks[int(_RNG.integers(0, 4)), int(r), int(c)] = False

    # Every output color is covered somewhere, which rejects hidden colors as
    # possible transparent colors.
    for color in map(int, np.unique(pattern)):
        coords = np.argwhere(pattern == color)
        r, c = coords[int(_RNG.integers(0, len(coords)))]
        view = int(_RNG.integers(0, 4))
        masks[view, int(r), int(c)] = True
        _clear_one_view_if_all_masked(masks, int(r), int(c), keep_masked=view)

    # Each view is the only visible witness at least once, so all four aligned
    # orientations can be necessary in a generated example.
    used: set[tuple[int, int]] = set()
    for visible_view in range(4):
        for _ in range(50):
            r = int(_RNG.integers(0, height))
            c = int(_RNG.integers(0, width))
            if (r, c) not in used:
                used.add((r, c))
                break
        masks[:, r, c] = True
        masks[visible_view, r, c] = False

    for view in range(4):
        if not np.any(masks[view]):
            r = int(_RNG.integers(0, height))
            c = int(_RNG.integers(0, width))
            masks[view, r, c] = True
            _clear_one_view_if_all_masked(masks, r, c, keep_masked=view)

    return masks


def _render_views(pattern: np.ndarray, masks: np.ndarray, hole_color: int) -> tuple[np.ndarray, ...]:
    ul = pattern.copy()
    ur = np.fliplr(pattern).copy()
    ll = np.flipud(pattern).copy()
    lr = np.flipud(np.fliplr(pattern)).copy()

    ul[masks[0]] = int(hole_color)
    ur[np.fliplr(masks[1])] = int(hole_color)
    ll[np.flipud(masks[2])] = int(hole_color)
    lr[np.flipud(np.fliplr(masks[3]))] = int(hole_color)
    return ul, ur, ll, lr


def _assemble_input(views: tuple[np.ndarray, ...], separator: int | None) -> np.ndarray:
    top = np.hstack((views[0], views[1]))
    bottom = np.hstack((views[2], views[3]))
    if separator is None:
        return np.vstack((top, bottom)).astype(int)

    height, width = map(int, views[0].shape)
    out = np.full((height * 2 + 1, width * 2 + 1), int(separator), dtype=int)
    out[:height, :width] = views[0]
    out[:height, width + 1 :] = views[1]
    out[height + 1 :, :width] = views[2]
    out[height + 1 :, width + 1 :] = views[3]
    return out


def _has_unaligned_conflict(views: tuple[np.ndarray, ...], hole_color: int) -> bool:
    stack = np.stack(views, axis=0)
    height, width = map(int, stack.shape[1:])
    for r in range(height):
        for c in range(width):
            values = {int(v) for v in stack[:, r, c] if int(v) != int(hole_color)}
            if len(values) > 1:
                return True
    return False


def _valid_generated(inp: np.ndarray, expected: np.ndarray, views: tuple[np.ndarray, ...], hole_color: int) -> bool:
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if expected.shape[0] < 4 or expected.shape[1] < 4:
        return False
    if np.array_equal(expected, np.fliplr(expected)):
        return False
    if np.array_equal(expected, np.flipud(expected)):
        return False
    if np.array_equal(expected, np.flipud(np.fliplr(expected))):
        return False
    if not _has_unaligned_conflict(views, hole_color):
        return False

    try:
        solved = main(inp)
    except AssertionError:
        return False
    return bool(np.array_equal(solved, expected))


def _gen_attempt() -> np.ndarray | None:
    height = int(_RNG.integers(4, 13))
    width = int(_RNG.integers(4, 13))
    use_separator = bool(_RNG.random() < 0.55)

    colors = list(_ARC_COLORS)
    _RNG.shuffle(colors)
    hole_color = int(colors.pop())
    separator = int(colors.pop()) if use_separator else None
    pattern_color_count = int(_RNG.integers(3, min(6, len(colors)) + 1))
    pattern_colors = [int(c) for c in colors[:pattern_color_count]]

    pattern = _make_hidden_pattern(height, width, pattern_colors)
    if len(set(map(int, pattern.flatten()))) < 3:
        return None

    masks = _make_masks(pattern)
    if np.any(np.all(masks, axis=0)):
        return None
    covered_colors = {int(pattern[r, c]) for r, c in np.argwhere(np.any(masks, axis=0))}
    if covered_colors != set(map(int, np.unique(pattern))):
        return None

    views = _render_views(pattern, masks, hole_color)
    if not all(np.any(view == hole_color) for view in views):
        return None

    inp = _assemble_input(views, separator)
    if _valid_generated(inp, pattern, views, hole_color):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid four-view occlusion input")
