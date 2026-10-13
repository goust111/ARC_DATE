from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
AZURE = 8
MARKER_COLORS = (BLUE, RED, GREEN, YELLOW, ORANGE, AZURE)
CORNER_ORDER = ("tl", "tr", "bl", "br")
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _rect_border(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    border: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))
    return border


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _extract_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for comp in _components(grid == GREY):
        coords = np.asarray(comp, dtype=int)
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "each frame must be at least 3x3"
        expected = _rect_border(r0, c0, r1, c1)
        actual = set(comp)
        assert actual == expected, "grey component must be a complete rectangular border"
        frames.append({"bbox": (r0, c0, r1, c1)})

    assert 2 <= len(frames) <= 4, "expected 2 to 4 concentric frames"
    frames.sort(key=lambda frame: (frame["bbox"][0], frame["bbox"][1], -frame["bbox"][2], -frame["bbox"][3]))

    for idx in range(1, len(frames)):
        prev = frames[idx - 1]["bbox"]
        cur = frames[idx]["bbox"]
        assert cur[0] == prev[0] + 2 and cur[1] == prev[1] + 2, "inner frame must be inset by 2 cells"
        assert cur[2] == prev[2] - 2 and cur[3] == prev[3] - 2, "inner frame must be inset by 2 cells"

    return frames


def _classify_marker(r: int, c: int, h: int, w: int) -> tuple[str, int]:
    candidates: list[tuple[str, int]] = []
    if r == c:
        candidates.append(("tl", r))
    if r == w - 1 - c:
        candidates.append(("tr", r))
    if h - 1 - r == c:
        candidates.append(("bl", c))
    if h - 1 - r == w - 1 - c:
        candidates.append(("br", h - 1 - r))

    assert candidates, "marker must lie on a corner diagonal"
    best_dist = min(dist for _corner, dist in candidates)
    best = [corner for corner, dist in candidates if dist == best_dist]
    assert len(best) == 1, "marker corner assignment must be unambiguous"
    return best[0], int(best_dist)


def _extract_corner_markers(grid: np.ndarray, n_frames: int) -> dict[str, list[tuple[int, int]]]:
    h, w = grid.shape
    marker_mask = (grid != BLACK) & (grid != GREY)
    comps = _components(marker_mask)
    assert len(comps) == 4 * n_frames, "expected one single-pixel marker for each corner/layer pair"
    assert all(len(comp) == 1 for comp in comps), "markers must be single pixels"

    grouped: dict[str, list[tuple[int, int]]] = {corner: [] for corner in CORNER_ORDER}
    for comp in comps:
        r, c = comp[0]
        corner, dist = _classify_marker(r, c, h, w)
        grouped[corner].append((dist, int(grid[r, c])))

    for corner in CORNER_ORDER:
        grouped[corner].sort(key=lambda item: item[0])
        assert len(grouped[corner]) == n_frames, "each corner must provide one marker per frame"
        dists = [dist for dist, _color in grouped[corner]]
        assert len(set(dists)) == n_frames, "marker distances in a corner must be distinct"

    return grouped


def _voted_color(colors: list[int]) -> int:
    counts = Counter(map(int, colors))
    best = max(counts.values())
    winners = [color for color, count in counts.items() if count == best]
    return int(winners[0]) if len(winners) == 1 else GREY


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each concentric grey frame from the majority vote of the equally ranked corner markers, then erase the markers.

    Input structure:
    - The background is black.
    - Near the grid center sit 2 to 4 one-pixel-thick grey rectangular frames nested concentrically, with exactly one black cell of spacing between neighboring frames.
    - In each corner region there is a diagonal chain of single-pixel colored markers that extends inward from the corner; each corner contributes one marker per frame layer.

    Rule:
    - Sort each corner's markers by increasing distance from that corner; the nearest markers correspond to the outermost frame and the farthest markers correspond to the innermost frame.
    - For every frame layer, gather the four equally ranked corner markers and take their color vote.
    - If one color has the unique highest count, recolor that whole frame with it; otherwise the layer stays grey.
    - Remove all corner markers to black and leave the black background and gaps between frames unchanged.

    Key insight:
    - The corner markers do not act locally: their distance rank encodes which concentric layer they control, and the layer color is decided only after combining the four corners' votes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _extract_frames(g)
    corner_markers = _extract_corner_markers(g, len(frames))

    out = np.zeros_like(g)
    for layer_idx, frame in enumerate(frames):
        votes = [corner_markers[corner][layer_idx][1] for corner in CORNER_ORDER]
        _paint_frame(out, frame["bbox"], _voted_color(votes))

    return out


def _corner_point(corner: str, dist: int, h: int, w: int) -> tuple[int, int]:
    if corner == "tl":
        return dist, dist
    if corner == "tr":
        return dist, w - 1 - dist
    if corner == "bl":
        return h - 1 - dist, dist
    if corner == "br":
        return h - 1 - dist, w - 1 - dist
    raise ValueError(corner)


def _balanced_layer_types(n_frames: int) -> list[str]:
    if n_frames % 2 == 0:
        tie_count = n_frames // 2
    else:
        tie_count = random.choice((n_frames // 2, n_frames // 2 + 1))
    layer_types = ["tie"] * tie_count + ["majority"] * (n_frames - tie_count)
    random.shuffle(layer_types)
    return layer_types


def _sample_layer_votes(layer_type: str) -> list[int]:
    if layer_type == "tie":
        a, b = random.sample(MARKER_COLORS, 2)
        votes = [a, a, b, b]
        random.shuffle(votes)
        return votes

    winner = random.choice(MARKER_COLORS)
    winner_count = random.choice((2, 3, 4))
    if winner_count == 2:
        others = random.sample([color for color in MARKER_COLORS if color != winner], 2)
        votes = [winner, winner, others[0], others[1]]
    elif winner_count == 3:
        other = random.choice([color for color in MARKER_COLORS if color != winner])
        votes = [winner, winner, winner, other]
    else:
        votes = [winner, winner, winner, winner]
    random.shuffle(votes)
    return votes


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.randint(2, 4)
    min_outer = 4 * n_frames - 1
    min_size = min_outer + 2 * (n_frames + 1)

    h = random.randint(min_size, 30)
    w = random.randint(min_size, 30)

    max_outer_h = h - 2 * (n_frames + 1)
    max_outer_w = w - 2 * (n_frames + 1)
    outer_h = random.randint(min_outer, max_outer_h)
    outer_w = random.randint(min_outer, max_outer_w)

    top = (h - outer_h) // 2
    left = (w - outer_w) // 2
    bottom = h - outer_h - top
    right = w - outer_w - left

    max_dists = {
        "tl": min(top, left) - 1,
        "tr": min(top, right) - 1,
        "bl": min(bottom, left) - 1,
        "br": min(bottom, right) - 1,
    }
    if any(max_dist < n_frames for max_dist in max_dists.values()):
        return None

    corner_dists = {
        corner: sorted(random.sample(range(1, max_dist + 1), n_frames))
        for corner, max_dist in max_dists.items()
    }

    layer_votes = [_sample_layer_votes(layer_type) for layer_type in _balanced_layer_types(n_frames)]

    inp = np.zeros((h, w), dtype=int)
    out = np.zeros_like(inp)

    for layer_idx in range(n_frames):
        bbox = (
            top + 2 * layer_idx,
            left + 2 * layer_idx,
            top + outer_h - 1 - 2 * layer_idx,
            left + outer_w - 1 - 2 * layer_idx,
        )
        _paint_frame(inp, bbox, GREY)
        _paint_frame(out, bbox, _voted_color(layer_votes[layer_idx]))

    for corner_idx, corner in enumerate(CORNER_ORDER):
        for layer_idx, dist in enumerate(corner_dists[corner]):
            r, c = _corner_point(corner, dist, h, w)
            if int(inp[r, c]) != BLACK:
                return None
            inp[r, c] = int(layer_votes[layer_idx][corner_idx])

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid concentric-frame voting instance")
