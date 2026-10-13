from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


BG = 0
MARKER = 9
FRAME_COLORS = (1, 2, 3, 7, 8)
MAX_GEN_ATTEMPTS = 500


@dataclass(frozen=True)
class Frame:
    r0: int
    c0: int
    r1: int
    c1: int
    color: int

    @property
    def height(self) -> int:
        return self.r1 - self.r0 + 1

    @property
    def width(self) -> int:
        return self.c1 - self.c0 + 1

    @property
    def area(self) -> int:
        return self.height * self.width

    def contains_interior(self, r: int, c: int) -> bool:
        return self.r0 < r < self.r1 and self.c0 < c < self.c1

    def border_cells(self) -> set[tuple[int, int]]:
        pts: set[tuple[int, int]] = set()
        for cc in range(self.c0, self.c1 + 1):
            pts.add((self.r0, cc))
            pts.add((self.r1, cc))
        for rr in range(self.r0, self.r1 + 1):
            pts.add((rr, self.c0))
            pts.add((rr, self.c1))
        return pts


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(grid == color):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            comp.append((rr, cc))
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        components.append(comp)
    return components


def _frame_from_component(grid: np.ndarray, color: int, comp: list[tuple[int, int]]) -> Frame:
    r0, c0, r1, c1 = _bbox(comp)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    assert height >= 3 and width >= 3, "frame must be at least 3x3"
    expected_area = 2 * (height + width) - 4
    assert len(comp) == expected_area, "component must be a 1-cell rectangular frame"
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(sub[0, :] == color)
    assert np.all(sub[-1, :] == color)
    assert np.all(sub[:, 0] == color)
    assert np.all(sub[:, -1] == color)
    assert not np.any(sub[1:-1, 1:-1] == color), "frame color may only appear on the border"
    return Frame(r0, c0, r1, c1, color)


def _extract_frames(grid: np.ndarray) -> list[Frame]:
    colors = [int(c) for c in np.unique(grid) if int(c) not in (BG, MARKER)]
    frames: list[Frame] = []
    for color in colors:
        for comp in _components_of_color(grid, color):
            frames.append(_frame_from_component(grid, color, comp))
    assert 3 <= len(frames) <= 4, "expected 3-4 nested frames"
    frames.sort(key=lambda fr: (-fr.area, fr.r0, fr.c0))
    for outer, inner in zip(frames, frames[1:]):
        assert outer.r0 < inner.r0 < inner.r1 < outer.r1
        assert outer.c0 < inner.c0 < inner.c1 < outer.c1
    return frames


def _containing_indices(frames: list[Frame], r: int, c: int) -> list[int]:
    return [idx for idx, frame in enumerate(frames) if frame.contains_interior(r, c)]


def _distances_to_interior_edges(frame: Frame, r: int, c: int) -> dict[str, int]:
    return {
        "left": c - (frame.c0 + 1),
        "right": (frame.c1 - 1) - c,
        "top": r - (frame.r0 + 1),
        "bottom": (frame.r1 - 1) - r,
    }


def _unique_side(frame: Frame, r: int, c: int) -> str:
    distances = _distances_to_interior_edges(frame, r, c)
    best = min(distances.values())
    sides = [side for side, value in distances.items() if value == best]
    assert len(sides) == 1, "marker side must be uniquely determined"
    return sides[0]


def _move_inward(r: int, c: int, side: str, steps: int) -> tuple[int, int]:
    if side == "left":
        return r, c + steps
    if side == "right":
        return r, c - steps
    if side == "top":
        return r + steps, c
    assert side == "bottom"
    return r - steps, c


def _on_any_frame_border(frames: list[Frame], r: int, c: int) -> bool:
    for frame in frames:
        if (r, c) in frame.border_cells():
            return True
    return False


def _marker_info(frames: list[Frame], r: int, c: int) -> dict[str, int | str | tuple[int, int]]:
    containing = _containing_indices(frames, r, c)
    depth = len(containing)
    assert depth > 0, "marker must be inside at least one frame interior"
    innermost_index = containing[-1]
    frame = frames[innermost_index]
    side = _unique_side(frame, r, c)
    target = _move_inward(r, c, side, depth)
    tr, tc = target
    assert frame.contains_interior(tr, tc), "projection must stay inside the innermost containing frame"
    assert not _on_any_frame_border(frames, tr, tc), "projection target may not land on a frame border"
    return {
        "depth": depth,
        "frame_index": innermost_index,
        "side": side,
        "target": target,
        "color": frame.color,
    }


def _simulate_with_rule(
    grid: np.ndarray,
    frames: list[Frame],
    marker_points: list[tuple[int, int]],
    *,
    distance_mode: str = "depth",
    color_mode: str = "innermost",
    frame_mode: str = "innermost",
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for r, c in sorted(marker_points):
        containing = _containing_indices(frames, r, c)
        depth = len(containing)
        assert depth > 0
        if frame_mode == "innermost":
            frame_index = containing[-1]
        else:
            frame_index = containing[0]
        frame = frames[frame_index]
        side = _unique_side(frame, r, c)
        steps = depth if distance_mode == "depth" else 1
        tr, tc = _move_inward(r, c, side, steps)
        out[r, c] = BG
        if 0 <= tr < out.shape[0] and 0 <= tc < out.shape[1] and frame.contains_interior(tr, tc):
            paint_color = frame.color if color_mode == "frame" else frames[containing[-1]].color
            if not _on_any_frame_border(frames, tr, tc):
                out[tr, tc] = paint_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each maroon marker inward by its nesting depth and recolor it with the innermost surrounding frame.

    Input structure:
    - The grid has a black background, 3-4 hollow rectangular frames of distinct non-maroon colors, and several maroon marker pixels.
    - The frames are strictly nested with black gaps between them, and each marker lies inside one or more frame interiors but never on a frame border.
    - Every marker is strictly closest to one side of its innermost containing frame.

    Rule:
    - Detect the nested rectangular frame components and sort them from outermost to innermost.
    - For each maroon marker in row-major order, count how many frame interiors contain it; this count is the projection distance.
    - Use the unique nearest side of the innermost containing frame to choose the inward direction, move the marker by that depth, paint the target with the innermost frame color, and erase the source marker.
    - Leave all frame pixels unchanged; if multiple projections hit the same target, the later row-major marker overwrites the earlier one.

    Key insight:
    - The containing-frame hierarchy is not only a classifier: the nesting depth itself becomes the exact pixel shift, while the innermost containing frame simultaneously determines direction and color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    frames = _extract_frames(g)
    marker_points = [(int(r), int(c)) for r, c in np.argwhere(g == MARKER)]
    assert marker_points, "expected at least one maroon marker"
    return _simulate_with_rule(g, frames, marker_points)


def _draw_frame(grid: np.ndarray, frame: Frame) -> None:
    grid[frame.r0, frame.c0 : frame.c1 + 1] = frame.color
    grid[frame.r1, frame.c0 : frame.c1 + 1] = frame.color
    grid[frame.r0 : frame.r1 + 1, frame.c0] = frame.color
    grid[frame.r0 : frame.r1 + 1, frame.c1] = frame.color


def _build_layout(rng: np.random.Generator) -> tuple[np.ndarray, list[Frame]]:
    layer_count = int(rng.integers(3, 5))
    margins = rng.integers(1, 4, size=4)
    gap_deltas = [int(rng.integers(3, 5)) for _ in range(layer_count - 1)]
    total_inset = sum(gap_deltas)
    inner_h = int(rng.integers(5, 10))
    inner_w = int(rng.integers(5, 12))
    outer_h = inner_h + 2 * total_inset
    outer_w = inner_w + 2 * total_inset
    h = int(margins[0] + outer_h + margins[1])
    w = int(margins[2] + outer_w + margins[3])
    if max(h, w) > 30:
        raise ValueError("layout too large")

    colors = [int(c) for c in rng.choice(np.array(FRAME_COLORS, dtype=int), size=layer_count, replace=False)]
    inner = Frame(
        int(margins[0] + total_inset),
        int(margins[2] + total_inset),
        int(margins[0] + total_inset + inner_h - 1),
        int(margins[2] + total_inset + inner_w - 1),
        colors[-1],
    )
    frames = [inner]
    for color, delta in zip(reversed(colors[:-1]), reversed(gap_deltas)):
        prev = frames[-1]
        frames.append(Frame(prev.r0 - delta, prev.c0 - delta, prev.r1 + delta, prev.c1 + delta, int(color)))
    frames.reverse()

    grid = np.zeros((h, w), dtype=int)
    for frame in frames:
        _draw_frame(grid, frame)
    return grid, frames


def _list_candidates(grid: np.ndarray, frames: list[Frame]) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    deepest_allowed = len(frames) - 1
    for r, c in np.argwhere(grid == BG):
        r = int(r)
        c = int(c)
        containing = _containing_indices(frames, r, c)
        depth = len(containing)
        if not (1 <= depth <= deepest_allowed):
            continue
        try:
            info = _marker_info(frames, r, c)
        except AssertionError:
            continue
        distances = _distances_to_interior_edges(frames[depth - 1], r, c)
        ordered = sorted(distances.values())
        if ordered[0] == ordered[1]:
            continue
        candidates.append(
            {
                "source": (r, c),
                "target": info["target"],
                "depth": info["depth"],
                "frame_index": info["frame_index"],
                "side": info["side"],
                "color": info["color"],
            }
        )
    return candidates


def _find_collision_pair(candidates: list[dict[str, object]]) -> list[dict[str, object]] | None:
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a = candidates[i]
            b = candidates[j]
            if a["target"] != b["target"]:
                continue
            if a["color"] == b["color"]:
                continue
            if a["source"] == b["source"]:
                continue
            return [a, b]
    return None


def _select_markers(
    rng: np.random.Generator,
    candidates: list[dict[str, object]],
) -> list[dict[str, object]] | None:
    shuffled = candidates.copy()
    rng.shuffle(shuffled)
    collision_pair = _find_collision_pair(shuffled)
    desire_collision = collision_pair is not None and bool(rng.integers(0, 100) < 35)

    for _ in range(300):
        marker_count = int(rng.integers(4, 7))
        selected: list[dict[str, object]] = []
        used_sources: set[tuple[int, int]] = set()
        used_targets: dict[tuple[int, int], int] = {}
        collision_target: tuple[int, int] | None = None
        allow_collision = desire_collision

        if allow_collision and collision_pair is not None:
            for cand in collision_pair:
                src = cand["source"]
                tgt = cand["target"]
                assert isinstance(src, tuple) and isinstance(tgt, tuple)
                selected.append(cand)
                used_sources.add(src)
                used_targets[tgt] = used_targets.get(tgt, 0) + 1
            collision_target = collision_pair[0]["target"]  # type: ignore[assignment]

        for cand in shuffled:
            if len(selected) >= marker_count:
                break
            src = cand["source"]
            tgt = cand["target"]
            assert isinstance(src, tuple) and isinstance(tgt, tuple)
            if src in used_sources:
                continue
            if tgt in used_sources or src in used_targets:
                continue
            if tgt in used_targets:
                if not allow_collision or tgt != collision_target or used_targets[tgt] >= 2:
                    continue
            selected.append(cand)
            used_sources.add(src)
            used_targets[tgt] = used_targets.get(tgt, 0) + 1

        if len(selected) != marker_count:
            continue

        depths = {int(cand["depth"]) for cand in selected}
        colors = {int(cand["color"]) for cand in selected}
        sides = {str(cand["side"]) for cand in selected}
        targets = [cand["target"] for cand in selected]
        if len(depths) < 2 or len(colors) < 2 or len(sides) < 2:
            continue
        if len(set(targets)) < marker_count - 1:
            continue
        if not any(int(cand["depth"]) > 1 for cand in selected):
            continue
        return sorted(selected, key=lambda item: item["source"])  # type: ignore[arg-type]

    return None


def _render_from_spec(grid: np.ndarray, markers: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for marker in sorted(markers, key=lambda item: item["source"]):
        sr, sc = marker["source"]  # type: ignore[misc]
        tr, tc = marker["target"]  # type: ignore[misc]
        out[sr, sc] = BG
        out[tr, tc] = int(marker["color"])
    return out


def _validate_scene(grid: np.ndarray, frames: list[Frame], markers: list[dict[str, object]]) -> bool:
    parsed = _extract_frames(grid)
    if [(f.r0, f.c0, f.r1, f.c1, f.color) for f in parsed] != [
        (f.r0, f.c0, f.r1, f.c1, f.color) for f in frames
    ]:
        return False

    expected = _render_from_spec(grid, markers)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, grid):
        return False

    marker_points = [marker["source"] for marker in markers]
    wrong_distance = _simulate_with_rule(grid, frames, marker_points, distance_mode="one")
    wrong_color = _simulate_with_rule(grid, frames, marker_points, color_mode="frame", frame_mode="outermost")
    if np.array_equal(expected, wrong_distance):
        return False
    if np.array_equal(expected, wrong_color):
        return False

    return True


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    try:
        grid, frames = _build_layout(rng)
    except ValueError:
        return None

    candidates = _list_candidates(grid, frames)
    if len(candidates) < 10:
        return None

    markers = _select_markers(rng, candidates)
    if markers is None:
        return None

    for marker in markers:
        sr, sc = marker["source"]  # type: ignore[misc]
        grid[sr, sc] = MARKER

    if not _validate_scene(grid, frames, markers):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-depth projection input")
