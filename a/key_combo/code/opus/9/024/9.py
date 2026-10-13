from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np


BG_COLOR = 0
FRAME_COLOR = 5
MARKER_COLORS = (1, 2, 3, 4)
CORNER_ORDER = ("tl", "tr", "bl", "br")
GEN_RETRIES = 400


@dataclass(frozen=True)
class FrameInfo:
    bbox: tuple[int, int, int, int]
    markers: dict[str, tuple[int, int]]
    colors: dict[str, int]
    has_majority: bool


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.asarray(coords, dtype=int))
    return parts


def _frame_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    return mask


def _marker_slots(bbox: tuple[int, int, int, int]) -> dict[str, tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {
        "tl": (r0 + 1, c0 + 1),
        "tr": (r0 + 1, c1 - 1),
        "bl": (r1 - 1, c0 + 1),
        "br": (r1 - 1, c1 - 1),
    }


def _destinations(info: FrameInfo) -> dict[str, tuple[int, int]]:
    r0, c0, r1, c1 = info.bbox
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    if info.has_majority:
        return {
            "tl": (r0 - 1, c0 - 1),
            "tr": (r0 - 1, c1 + 1),
            "bl": (r1 + 1, c0 - 1),
            "br": (r1 + 1, c1 + 1),
        }
    return {
        "tl": (r0 - 1, mid_c),
        "tr": (mid_r, c1 + 1),
        "bl": (mid_r, c0 - 1),
        "br": (r1 + 1, mid_c),
    }


def _extract_frames(grid: np.ndarray) -> list[FrameInfo]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    frame_parts = _components(g == FRAME_COLOR)
    assert 2 <= len(frame_parts) <= 3, "expected 2 or 3 grey rectangular frames"

    frame_infos: list[FrameInfo] = []
    used_marker_positions: set[tuple[int, int]] = set()
    expected_frame_cells = np.zeros(g.shape, dtype=bool)

    for coords in frame_parts:
        bbox = _bbox(coords)
        r0, c0, r1, c1 = bbox
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 5 and w >= 5, "frame too small"
        assert h % 2 == 1 and w % 2 == 1, "frame dimensions must be odd for unique side midpoints"
        assert r0 >= 1 and c0 >= 1 and r1 + 1 < g.shape[0] and c1 + 1 < g.shape[1], (
            "frame needs one-cell outer margin"
        )

        mask = _frame_mask(g.shape, bbox)
        comp_mask = np.zeros(g.shape, dtype=bool)
        comp_mask[coords[:, 0], coords[:, 1]] = True
        assert np.array_equal(mask, comp_mask), "grey component must be a 1-pixel-thick hollow rectangle"
        expected_frame_cells |= mask

        markers = _marker_slots(bbox)
        colors: dict[str, int] = {}
        for name, pos in markers.items():
            color = int(g[pos])
            assert color not in {BG_COLOR, FRAME_COLOR}, f"missing colored marker at {name}"
            colors[name] = color
            used_marker_positions.add(pos)

        counts = Counter(colors.values())
        has_majority = counts.most_common(1)[0][1] >= 3
        info = FrameInfo(bbox=bbox, markers=markers, colors=colors, has_majority=has_majority)
        frame_infos.append(info)

    non_bg = g != BG_COLOR
    expected_non_bg = expected_frame_cells.copy()
    for pos in used_marker_positions:
        expected_non_bg[pos] = True
    assert np.array_equal(non_bg, expected_non_bg), "unexpected extra foreground pixels"

    all_dests: set[tuple[int, int]] = set()
    for info in frame_infos:
        for dst in _destinations(info).values():
            assert _in_bounds(g.shape, *dst), "destination fell outside the grid"
            assert int(g[dst]) == BG_COLOR, "destination cell must start as background"
            all_dests.add(dst)
    assert len(all_dests) == 4 * len(frame_infos), "destination slots must be unique"

    frame_infos.sort(key=lambda info: info.bbox)
    return frame_infos


def main(grid: np.ndarray) -> np.ndarray:
    """Move each frame's four inner corner markers outward according to whether their colors have a majority.

    Input structure:
    - The background is black and contains 2 or 3 separate one-pixel-thick grey hollow rectangles.
    - Each rectangle has exactly four colored single-pixel markers, one at each interior corner slot.
    - Each marker keeps its own color; only the four marker colors on the same rectangle are compared.

    Rule:
    - For each rectangle, count the colors of its four interior-corner markers.
    - If some color appears at least 3 times, erase the four inner markers and move them to the matching outer corners one cell beyond the frame.
    - Otherwise erase the four inner markers and move them to the four outside side-midpoint slots in the fixed mapping: top-left to top midpoint, top-right to right midpoint, bottom-left to left midpoint, bottom-right to bottom midpoint.

    Key insight:
    - The same four visible corner markers serve two roles at once: their positions identify which marker is which, while their within-rectangle color multiplicities decide whether the destinations are radial outer corners or side midpoints.
    """
    g = np.asarray(grid, dtype=int)
    infos = _extract_frames(g)
    out = g.copy()
    for info in infos:
        dests = _destinations(info)
        for name in CORNER_ORDER:
            src = info.markers[name]
            out[src] = BG_COLOR
            out[dests[name]] = info.colors[name]
    return out


def _gap_at_least(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int], gap: int) -> bool:
    r0, c0, r1, c1 = box_a
    rr0, cc0, rr1, cc1 = box_b
    return r1 + gap < rr0 or rr1 + gap < r0 or c1 + gap < cc0 or cc1 + gap < c0


def _sample_box(rng: np.random.Generator, shape: tuple[int, int], boxes: Iterable[tuple[int, int, int, int]]):
    h, w = shape
    for _ in range(200):
        box_h = int(rng.choice([5, 7, 9]))
        box_w = int(rng.choice([5, 7, 9]))
        if box_h + 2 > h or box_w + 2 > w:
            continue
        r0 = int(rng.integers(1, h - box_h))
        c0 = int(rng.integers(1, w - box_w))
        box = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
        if all(_gap_at_least(box, other, 3) for other in boxes):
            return box
    return None


def _majority_colors(rng: np.random.Generator) -> list[int]:
    major = int(rng.choice(MARKER_COLORS))
    if rng.random() < 0.25:
        return [major, major, major, major]
    other = int(rng.choice([c for c in MARKER_COLORS if c != major]))
    colors = [major, major, major, other]
    rng.shuffle(colors)
    return colors


def _non_majority_colors(rng: np.random.Generator) -> list[int]:
    if rng.random() < 0.5:
        a, b = map(int, rng.choice(MARKER_COLORS, size=2, replace=False))
        colors = [a, a, b, b]
        rng.shuffle(colors)
        return colors
    return [int(c) for c in rng.permutation(MARKER_COLORS)]


def _box_targets(box: tuple[int, int, int, int], majority: bool) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    if majority:
        return {(r0 - 1, c0 - 1), (r0 - 1, c1 + 1), (r1 + 1, c0 - 1), (r1 + 1, c1 + 1)}
    return {(r0 - 1, mid_c), (mid_r, c1 + 1), (mid_r, c0 - 1), (r1 + 1, mid_c)}


def _valid_layout(boxes: list[tuple[int, int, int, int]], types: list[bool]) -> bool:
    all_targets: set[tuple[int, int]] = set()
    for box, majority in zip(boxes, types):
        targets = _box_targets(box, majority)
        if len(targets) != 4 or all_targets & targets:
            return False
        all_targets |= targets
    return True


def _semantic_check(inp: np.ndarray) -> None:
    infos = _extract_frames(inp)
    assert any(info.has_majority for info in infos), "need at least one majority rectangle"
    assert any(not info.has_majority for info in infos), "need at least one non-majority rectangle"

    signatures = [tuple(sorted(Counter(info.colors.values()).items())) for info in infos]
    assert len(set(signatures)) == len(signatures), "rectangles should not reuse the same color-count pattern"

    for i, a in enumerate(infos):
        for b in infos[i + 1 :]:
            assert _gap_at_least(a.bbox, b.bbox, 3), "frames too close to each other"

    out = main(inp)
    changed = {tuple(map(int, rc)) for rc in np.argwhere(inp != out)}
    expected_changed: set[tuple[int, int]] = set()

    for info in infos:
        dests = _destinations(info)
        for name in CORNER_ORDER:
            src = info.markers[name]
            dst = dests[name]
            assert int(out[src]) == BG_COLOR, "source marker must be erased"
            assert int(out[dst]) == info.colors[name], "marker must land in its prescribed destination"
            expected_changed.add(src)
            expected_changed.add(dst)
        frame_mask = _frame_mask(inp.shape, info.bbox)
        assert np.array_equal(inp[frame_mask], out[frame_mask]), "frame border must stay unchanged"

    assert changed == expected_changed, "only source and destination slots may change"


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_frames = int(rng.integers(2, 4))
    min_side = 20 if n_frames == 3 else 18
    shape = (int(rng.integers(min_side, 25)), int(rng.integers(min_side, 25)))

    frame_types = [True, False]
    if n_frames == 3:
        frame_types.append(bool(rng.integers(0, 2)))
    rng.shuffle(frame_types)

    boxes: list[tuple[int, int, int, int]] = []
    for _ in range(n_frames):
        box = _sample_box(rng, shape, boxes)
        if box is None:
            return None
        boxes.append(box)
    if not _valid_layout(boxes, frame_types):
        return None

    color_lists: list[list[int]] = []
    seen_signatures = set()
    for majority in frame_types:
        for _ in range(50):
            colors = _majority_colors(rng) if majority else _non_majority_colors(rng)
            signature = tuple(sorted(Counter(colors).items()))
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                color_lists.append(colors)
                break
        else:
            return None

    grid = np.full(shape, BG_COLOR, dtype=int)
    for box, colors in zip(boxes, color_lists):
        r0, c0, r1, c1 = box
        grid[r0 : r1 + 1, c0] = FRAME_COLOR
        grid[r0 : r1 + 1, c1] = FRAME_COLOR
        grid[r0, c0 : c1 + 1] = FRAME_COLOR
        grid[r1, c0 : c1 + 1] = FRAME_COLOR
        for name, color in zip(CORNER_ORDER, colors):
            grid[_marker_slots(box)[name]] = int(color)

    _semantic_check(grid)
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
