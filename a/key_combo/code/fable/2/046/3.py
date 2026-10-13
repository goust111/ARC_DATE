from __future__ import annotations

from collections import Counter
import random
from typing import Callable

import numpy as np


PANEL_H = 5
GENERATION_ATTEMPTS = 2000


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste_mask(dst: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    r0 = int(r0)
    c0 = int(c0)
    if r0 < 0 or c0 < 0 or r0 + h > dst.shape[0] or c0 + w > dst.shape[1]:
        raise ValueError("mask placement is out of bounds")
    view = dst[r0 : r0 + h, c0 : c0 + w]
    view[mask.astype(bool)] = int(color)


TRANSFORMS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "I": lambda a: a.copy(),
    "R90": lambda a: np.rot90(a, 1).copy(),
    "R180": lambda a: np.rot90(a, 2).copy(),
    "R270": lambda a: np.rot90(a, 3).copy(),
    "FH": lambda a: np.fliplr(a).copy(),
    "FV": lambda a: np.flipud(a).copy(),
    "FD": lambda a: a.T.copy(),
    "FA": lambda a: np.fliplr(np.flipud(a)).T.copy(),
}

PAIR_FAMILIES = (
    ("FD", ("R270", "FH")),
    ("FD", ("R90", "FV")),
    ("FA", ("R90", "FH")),
    ("FA", ("R270", "FV")),
)


def _apply_transform(name: str, patch: np.ndarray) -> np.ndarray:
    return TRANSFORMS[name](patch)


def _split_panels(grid: np.ndarray) -> tuple[list[np.ndarray], int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    sep_rows = [r for r in range(g.shape[0]) if np.all(g[r] == g[r, 0]) and int(g[r, 0]) != int(bg)]
    assert len(sep_rows) == 4, "expected exactly four separator rows"
    sep_color = int(g[sep_rows[0], 0])
    assert all(int(g[r, 0]) == sep_color for r in sep_rows), "separator rows must share one color"

    cuts = [-1] + sep_rows + [g.shape[0]]
    panels = [g[cuts[i] + 1 : cuts[i + 1]].copy() for i in range(len(cuts) - 1)]
    assert len(panels) == 5, "expected exactly five panels"
    heights = {panel.shape[0] for panel in panels}
    widths = {panel.shape[1] for panel in panels}
    assert len(heights) == 1 and len(widths) == 1, "panels must be equal-sized"
    return panels, bg, sep_color


def _extract_sprite(panel: np.ndarray, bg: int) -> dict[str, object] | None:
    colors = sorted(set(map(int, panel.ravel())) - {int(bg)})
    if not colors:
        return None
    assert len(colors) == 1, "each visible panel must contain one monochrome sprite"
    color = int(colors[0])
    mask = panel == color
    box = _bbox(mask)
    assert box is not None
    return {
        "color": color,
        "bbox": box,
        "patch": _crop_bbox(mask, box),
    }


def _step_candidates(src_patch: np.ndarray, dst_patch: np.ndarray) -> list[str]:
    matches = []
    for name in TRANSFORMS:
        pred = _apply_transform(name, src_patch)
        if pred.shape == dst_patch.shape and np.array_equal(pred, dst_patch):
            matches.append(name)
    return matches


def _solve_panels(panels: list[np.ndarray], bg: int) -> np.ndarray:
    assert len(panels) == 5
    assert np.all(panels[2] == bg), "frame 3 must be blank in the input"

    info1 = _extract_sprite(panels[0], bg)
    info2 = _extract_sprite(panels[1], bg)
    info4 = _extract_sprite(panels[3], bg)
    info5 = _extract_sprite(panels[4], bg)
    assert all(info is not None for info in (info1, info2, info4, info5))

    b1 = info1["bbox"]
    b2 = info2["bbox"]
    b4 = info4["bbox"]
    b5 = info5["bbox"]
    p1 = info1["patch"]
    p2 = info2["patch"]
    p4 = info4["patch"]
    p5 = info5["patch"]
    color = int(info2["color"])
    assert int(info1["color"]) == color == int(info4["color"]) == int(info5["color"])

    delta = (int(b2[0] - b1[0]), int(b2[1] - b1[1]))
    assert delta != (0, 0), "translation offset must be visible"
    assert (int(b5[0] - b4[0]), int(b5[1] - b4[1])) == delta, "the visible single-step offset must be constant"
    assert (int(b4[0] - b2[0]), int(b4[1] - b2[1])) == (2 * delta[0], 2 * delta[1]), (
        "frame 2 to frame 4 must span exactly two translation steps"
    )

    candidates = _step_candidates(p1, p2)
    assert len(candidates) >= 2, "single-step evidence must be ambiguous"

    winners = []
    for name in candidates:
        if np.array_equal(_apply_transform(name, _apply_transform(name, p2)), p4):
            winners.append(name)
    assert len(winners) == 1, "the two-step relation must determine a unique transform"
    transform = winners[0]

    assert np.array_equal(_apply_transform(transform, p4), p5), "frame 5 must agree with the recovered transform"

    out = np.full_like(panels[2], int(bg))
    patch3 = _apply_transform(transform, p2)
    _paste_mask(out, patch3, int(b2[0] + delta[0]), int(b2[1] + delta[1]), color)
    return out


def _is_contiguous(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _symmetrize(base: np.ndarray, diag_type: str) -> np.ndarray:
    out = base.copy()
    n = base.shape[0]
    for r in range(n):
        for c in range(n):
            if not base[r, c]:
                continue
            if diag_type == "FD":
                out[c, r] = True
            else:
                out[n - 1 - c, n - 1 - r] = True
    return out


def _random_ambiguous_sprite(size: int, diag_type: str) -> np.ndarray | None:
    for _ in range(1000):
        base = np.zeros((size, size), dtype=bool)
        for r in range(size):
            for c in range(size):
                if diag_type == "FD" and r > c:
                    continue
                if diag_type == "FA" and r + c > size - 1:
                    continue
                if random.random() < 0.45:
                    base[r, c] = True

        a = random.randrange(size)
        b = random.randrange(size)
        if diag_type == "FD":
            base[0, a] = True
            base[a, 0] = True
            base[size - 1, b] = True
            base[b, size - 1] = True
        else:
            base[0, a] = True
            base[size - 1 - a, size - 1] = True
            base[size - 1, b] = True
            base[size - 1 - b, 0] = True

        mask = _symmetrize(base, diag_type)
        if int(mask.sum()) < size + 2 or int(mask.sum()) > size * size - 2:
            continue
        if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if not _is_contiguous(mask):
            continue

        stabilizers = [name for name in TRANSFORMS if np.array_equal(_apply_transform(name, mask), mask)]
        if set(stabilizers) != {"I", diag_type}:
            continue
        return mask
    return None


def _stack_panels(panels: list[np.ndarray], sep_color: int) -> np.ndarray:
    rows = []
    for i, panel in enumerate(panels):
        rows.append(panel)
        if i != len(panels) - 1:
            rows.append(np.full((1, panel.shape[1]), int(sep_color), dtype=int))
    return np.vstack(rows)


def _validate_semantics(grid: np.ndarray, true_transform: str, expected: np.ndarray) -> bool:
    panels, bg, _sep = _split_panels(grid)
    if not np.all(panels[2] == bg):
        return False

    info1 = _extract_sprite(panels[0], bg)
    info2 = _extract_sprite(panels[1], bg)
    info4 = _extract_sprite(panels[3], bg)
    info5 = _extract_sprite(panels[4], bg)
    if any(info is None for info in (info1, info2, info4, info5)):
        return False

    c12 = _step_candidates(info1["patch"], info2["patch"])
    c45 = _step_candidates(info4["patch"], info5["patch"])
    if len(c12) != 2 or len(c45) != 2:
        return False
    if true_transform not in c12 or true_transform not in c45:
        return False

    winners = [
        name
        for name in c12
        if np.array_equal(_apply_transform(name, _apply_transform(name, info2["patch"])), info4["patch"])
    ]
    if winners != [true_transform]:
        return False

    solved = _solve_panels(panels, bg)
    return np.array_equal(solved, expected)


def _gen_attempt() -> np.ndarray | None:
    bg, sprite_color, sep_color = random.sample(range(10), 3)
    size = random.choice((3, 4))
    diag_type, pair = random.choice(PAIR_FAMILIES)
    true_transform = random.choice(pair)

    sprite = _random_ambiguous_sprite(size, diag_type)
    if sprite is None:
        return None

    dc = random.choice((-3, -2, -1, 1, 2, 3))
    min_w = size + 4 * abs(dc) + 2
    if min_w > 30:
        return None
    panel_w = random.randint(max(14, min_w), 26)

    row0 = 1 if size == 4 else random.choice((1, 2))
    c_low = 1 + max(0, -4 * dc)
    c_high = panel_w - size - 1 - max(0, 4 * dc)
    if c_low > c_high:
        return None
    c0 = random.randint(c_low, c_high)

    patches = [sprite]
    for _ in range(4):
        patches.append(_apply_transform(true_transform, patches[-1]))

    panels = [np.full((PANEL_H, panel_w), int(bg), dtype=int) for _ in range(5)]
    for idx in (0, 1, 3, 4):
        _paste_mask(panels[idx], patches[idx], row0, c0 + idx * dc, sprite_color)

    expected = np.full_like(panels[2], int(bg))
    _paste_mask(expected, patches[2], row0, c0 + 2 * dc, sprite_color)

    grid = _stack_panels(panels, sep_color)
    if not _validate_semantics(grid, true_transform, expected):
        return None
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the blank middle panel of a five-frame sprite animation.

    Input structure:
    - The input is a vertical stack of five equal-sized panels separated by
      solid full-width separator rows.
    - Panels 1, 2, 4, and 5 each contain one monochrome sprite on a shared
      background, while panel 3 is entirely blank.
    - The sprite keeps the same color and moves by a fixed bbox translation
      each step.

    Rule:
    - Between consecutive frames, apply one unknown orthogonal transform
      (rotation or reflection) to the sprite patch, then translate its bbox by
      the same fixed offset.
    - Use the ambiguous one-step relation from frame 1 to frame 2 to collect
      candidate transforms, then compare frame 2 with frame 4 to see which
      candidate's two-step composite matches the observed shape two steps later.
    - Apply that unique transform once to frame 2 and place the result one
      translation step later to produce the missing frame 3.

    Key insight:
    The sprite is intentionally chosen so a single observed step admits two
    orthogonal transform hypotheses, but the two-step composite from frame 2 to
    frame 4 distinguishes identity-vs-180 outcomes and makes the middle frame
    unique.
    """
    panels, bg, _sep = _split_panels(np.asarray(grid, dtype=int))
    return _solve_panels(panels, bg)


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            assert grid.ndim == 2 and max(grid.shape) <= 30
            return grid
    raise RuntimeError("failed to generate a valid ambiguous-middle-frame input")
