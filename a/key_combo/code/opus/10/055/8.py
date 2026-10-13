from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0
FRAME_SIDE_RANGE = (5, 7)
TARGET_GRID_RANGE = (18, 30)
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _color_set(inner: np.ndarray) -> tuple[int, ...]:
    return tuple(sorted(int(v) for v in np.unique(inner) if int(v) != BACKGROUND))


def _dominant_color(column: np.ndarray) -> int:
    vals = [int(v) for v in column if int(v) != BACKGROUND]
    assert vals, "every interior column must contain a non-black pixel"
    ranked = Counter(vals).most_common()
    if len(ranked) > 1:
        assert ranked[0][1] > ranked[1][1], "column dominant color must be unique"
    return int(ranked[0][0])


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frames: list[tuple[int, int, int, int, int]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        coords = np.argwhere(g == color)
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        side_h = int(r1 - r0 + 1)
        side_w = int(c1 - c0 + 1)
        if side_h != side_w or not (FRAME_SIDE_RANGE[0] <= side_h <= FRAME_SIDE_RANGE[1]):
            continue
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros_like(patch, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.all(patch[border] == color):
            continue
        if np.any(patch[~border] == color):
            continue
        if int(np.sum(g == color)) != int(border.sum()):
            continue
        frames.append((int(r0), int(c0), int(r1), int(c1), int(color)))
    frames.sort()
    return frames


def _extract_frames(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = _find_frames(grid)
    assert len(frames) == 3, "expected exactly three hollow square frames"

    inners = []
    for r0, c0, r1, c1, _color in frames:
        inners.append(np.asarray(grid[r0 + 1 : r1, c0 + 1 : c1], dtype=int).copy())

    signatures = [_color_set(inner) for inner in inners]
    pair: tuple[int, int] | None = None
    for i in range(3):
        for j in range(i + 1, 3):
            if signatures[i] == signatures[j]:
                assert pair is None, "training pair must be unique"
                pair = (i, j)
    assert pair is not None, "expected one same-palette training pair"

    test_idx = ({0, 1, 2} - set(pair)).pop()
    a, b = pair
    count_a = int(np.count_nonzero(inners[a]))
    count_b = int(np.count_nonzero(inners[b]))
    assert count_a != count_b, "training source/destination must be distinguishable by density"
    src_idx, dst_idx = (a, b) if count_a > count_b else (b, a)
    return inners[src_idx], inners[dst_idx], inners[test_idx]


def _analog_mapping(src: np.ndarray, dst: np.ndarray, test: np.ndarray) -> dict[int, int]:
    assert src.shape == dst.shape == test.shape
    width = int(src.shape[1])
    src_dom = [_dominant_color(src[:, c]) for c in range(width)]
    dst_dom = [_dominant_color(dst[:, c]) for c in range(width)]
    test_dom = [_dominant_color(test[:, c]) for c in range(width)]

    train_perm: dict[int, int] = {}
    for s_col, d_col in zip(src_dom, dst_dom):
        old = train_perm.get(int(s_col))
        if old is not None:
            assert old == int(d_col), "training mapping must be globally consistent"
        train_perm[int(s_col)] = int(d_col)

    role_to_test: dict[int, int] = {}
    for role in sorted(set(src_dom)):
        values = {int(test_dom[j]) for j in range(width) if int(src_dom[j]) == int(role)}
        assert len(values) == 1, "test palette transfer must be determined by column roles"
        role_to_test[int(role)] = int(next(iter(values)))

    mapped: dict[int, int] = {}
    for src_role, dst_role in train_perm.items():
        mapped[int(role_to_test[src_role])] = int(role_to_test[dst_role])
    return mapped


def _apply_mapping(inner: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = np.asarray(inner, dtype=int).copy()
    original = out.copy()
    for src_color, dst_color in mapping.items():
        out[original == int(src_color)] = int(dst_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the training pair's column-role recoloring rule and transfer it to the test frame.

    Input structure:
    - The background is black.
    - The grid contains exactly three equal-sized hollow square frames with distinct border colors.
    - Each frame interior is a small sprite whose columns all contain at least one non-black pixel; two interiors share the same non-black color set and form the training pair, while the third uses a different palette and is the test input.

    Rule:
    - Identify the training pair by matching interior color sets, then choose the denser one as the training source and the sparser one as the training destination.
    - For each interior column, take the unique most frequent non-black color in the source and destination; these dominant colors define a consistent color-role mapping from source palette roles to destination palette roles.
    - Transfer that role mapping to the test frame by matching test-column dominant colors against the source columns at the same positions, then recolor every mapped test color accordingly while leaving black and any unmapped accent colors unchanged.

    Key insight:
    - The useful signal is not the literal training colors but the per-column role sequence: the destination shows how each source role should change, and the test reuses the same role sequence with different actual colors.
    """
    src, dst, test = _extract_frames(np.asarray(grid, dtype=int))
    mapping = _analog_mapping(src, dst, test)
    return _apply_mapping(test, mapping)


def _sample_role_sequence(width: int, role_count: int) -> list[int]:
    roles = list(range(role_count))
    seq = roles[:]
    while len(seq) < width:
        seq.append(random.randrange(role_count))
    random.shuffle(seq)
    return [int(v) for v in seq]


def _random_derangement(values: list[int]) -> dict[int, int]:
    src = list(values)
    dst = src[:]
    while True:
        random.shuffle(dst)
        if all(a != b for a, b in zip(src, dst)):
            return {int(a): int(b) for a, b in zip(src, dst)}


def _segment_rows(size: int, count: int) -> list[int]:
    start = random.randint(0, size - count)
    return list(range(start, start + count))


def _build_interior(
    size: int,
    role_seq: list[int],
    role_colors: list[int],
    mode: str,
    accent_color: int | None = None,
) -> np.ndarray:
    inner = np.zeros((size, size), dtype=int)
    for col, role in enumerate(role_seq):
        dominant = int(role_colors[role])
        if mode == "source":
            dom_count = random.randint(max(2, size // 2 + 1), size)
        elif mode == "dest":
            dom_count = random.randint(1, max(1, size // 2))
        elif mode == "test":
            dom_count = random.randint(max(1, size // 2), size)
        else:
            raise ValueError(mode)

        rows = _segment_rows(size, dom_count)
        inner[rows, col] = dominant

        other_choices = [int(c) for c in role_colors if int(c) != dominant]
        if accent_color is not None:
            other_choices.append(int(accent_color))
        empty_rows = [r for r in range(size) if inner[r, col] == BACKGROUND]
        if dom_count >= 2 and empty_rows and other_choices and random.random() < 0.6:
            minor_row = random.choice(empty_rows)
            inner[minor_row, col] = random.choice(other_choices)
    return inner


def _frame_patch(inner: np.ndarray, frame_color: int) -> np.ndarray:
    size = int(inner.shape[0]) + 2
    patch = np.zeros((size, size), dtype=int)
    patch[0, :] = int(frame_color)
    patch[-1, :] = int(frame_color)
    patch[:, 0] = int(frame_color)
    patch[:, -1] = int(frame_color)
    patch[1:-1, 1:-1] = inner
    return patch


def _layout_positions(side: int) -> tuple[int, int, list[tuple[int, int]]]:
    g1 = random.randint(1, 3)
    g2 = random.randint(1, 3)
    layout = random.choice(["row", "col", "L_left", "L_right"])
    if layout == "row":
        positions = [(0, 0), (0, side + g1), (0, 2 * side + g1 + g2)]
    elif layout == "col":
        positions = [(0, 0), (side + g1, 0), (2 * side + g1 + g2, 0)]
    elif layout == "L_left":
        positions = [(0, 0), (0, side + g1), (side + g2, 0)]
    else:
        positions = [(0, 0), (0, side + g1), (side + g2, side + g1)]

    max_r = max(r for r, _c in positions) + side
    max_c = max(c for _r, c in positions) + side
    final_h = random.randint(max(TARGET_GRID_RANGE[0], max_r), TARGET_GRID_RANGE[1])
    final_w = random.randint(max(TARGET_GRID_RANGE[0], max_c), TARGET_GRID_RANGE[1])
    shift_r = random.randint(0, final_h - max_r)
    shift_c = random.randint(0, final_w - max_c)
    shifted = [(r + shift_r, c + shift_c) for r, c in positions]
    return final_h, final_w, shifted


def _expected_test_output(test_inner: np.ndarray, test_colors: list[int], train_perm: dict[int, int], train_colors: list[int]) -> np.ndarray:
    role_to_test = {int(train_colors[i]): int(test_colors[i]) for i in range(len(train_colors))}
    test_mapping = {
        int(role_to_test[src_role]): int(role_to_test[dst_role])
        for src_role, dst_role in train_perm.items()
    }
    return _apply_mapping(test_inner, test_mapping)


def _render_case() -> tuple[np.ndarray, np.ndarray]:
    side = random.randint(*FRAME_SIDE_RANGE)
    inner_size = side - 2
    role_count = 3 if random.random() < 0.75 else 2

    colors = list(range(1, 10))
    random.shuffle(colors)
    frame_colors = colors[:3]
    offset = 3
    train_colors = colors[offset : offset + role_count]
    offset += role_count
    test_colors = colors[offset : offset + role_count]
    offset += role_count
    accent_color = colors[offset] if offset < len(colors) and role_count == 2 and random.random() < 0.6 else None

    role_seq = _sample_role_sequence(inner_size, role_count)
    train_perm = _random_derangement(train_colors)

    src_inner = _build_interior(inner_size, role_seq, train_colors, "source")
    dst_inner = _build_interior(inner_size, role_seq, [train_perm[c] for c in train_colors], "dest")
    test_inner = _build_interior(inner_size, role_seq, test_colors, "test", accent_color=accent_color)
    expected = _expected_test_output(test_inner, test_colors, train_perm, train_colors)

    assert int(np.count_nonzero(src_inner)) > int(np.count_nonzero(dst_inner))
    assert _color_set(src_inner) == _color_set(dst_inner) == tuple(sorted(train_colors))
    test_sig = _color_set(test_inner)
    assert tuple(sorted(test_colors)) == tuple(sorted(set(test_sig) & set(test_colors)))
    assert test_sig != tuple(sorted(train_colors))
    assert not np.array_equal(expected, test_inner)

    patches = [
        _frame_patch(src_inner, frame_colors[0]),
        _frame_patch(dst_inner, frame_colors[1]),
        _frame_patch(test_inner, frame_colors[2]),
    ]
    order = [0, 1, 2]
    random.shuffle(order)

    grid_h, grid_w, positions = _layout_positions(side)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    for idx, (r0, c0) in zip(order, positions):
        patch = patches[idx]
        h, w = patch.shape
        assert np.all(grid[r0 : r0 + h, c0 : c0 + w] == BACKGROUND)
        grid[r0 : r0 + h, c0 : c0 + w] = patch

    return grid, expected


def _semantic_ok(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        src, dst, test = _extract_frames(grid)
        mapping = _analog_mapping(src, dst, test)
        actual = _apply_mapping(test, mapping)
    except AssertionError:
        return False

    if not np.array_equal(actual, expected):
        return False

    direct_train = _apply_mapping(test, {k: v for k, v in _analog_mapping(src, dst, src).items() if k in np.unique(test)})
    if np.array_equal(direct_train, expected):
        return False

    frames = _find_frames(grid)
    if len(frames) != 3:
        return False

    sigs = [_color_set(grid[r0 + 1 : r1, c0 + 1 : c1]) for r0, c0, r1, c1, _ in frames]
    if sum(sigs[i] == sigs[j] for i in range(3) for j in range(i + 1, 3)) != 1:
        return False

    width = int(src.shape[1])
    src_dom = [_dominant_color(src[:, c]) for c in range(width)]
    test_dom = [_dominant_color(test[:, c]) for c in range(width)]
    for role in set(src_dom):
        if len({test_dom[j] for j in range(width) if src_dom[j] == role}) != 1:
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid, expected = _render_case()
        if _semantic_ok(grid, expected):
            return grid
    raise RuntimeError("failed to generate a valid analogy recoloring sample")
