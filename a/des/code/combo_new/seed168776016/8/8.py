from __future__ import annotations

import numpy as np


BG = 0
GREY = 5
FRAME_COLORS = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
MAX_GENERATION_ATTEMPTS = 400
PAIR_PLACEMENT_ATTEMPTS = 200
PATTERN_ATTEMPTS = 200


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if visited[r, c]:
            continue
        stack = [(r, c)]
        visited[r, c] = True
        pts: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            pts.append((rr, cc))
            if rr > 0 and mask[rr - 1, cc] and not visited[rr - 1, cc]:
                visited[rr - 1, cc] = True
                stack.append((rr - 1, cc))
            if rr + 1 < h and mask[rr + 1, cc] and not visited[rr + 1, cc]:
                visited[rr + 1, cc] = True
                stack.append((rr + 1, cc))
            if cc > 0 and mask[rr, cc - 1] and not visited[rr, cc - 1]:
                visited[rr, cc - 1] = True
                stack.append((rr, cc - 1))
            if cc + 1 < w and mask[rr, cc + 1] and not visited[rr, cc + 1]:
                visited[rr, cc + 1] = True
                stack.append((rr, cc + 1))
        comps.append(np.array(pts, dtype=int))
    return comps


def _is_rect_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return False
    local = np.zeros((h, w), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return np.array_equal(local, expected)


def _find_frame_pairs(grid: np.ndarray, bg: int) -> dict[int, list[dict[str, tuple[int, int, int, int] | int]]]:
    pairs: dict[int, list[dict[str, tuple[int, int, int, int] | int]]] = {}
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        frames = []
        for coords in _components(grid == color):
            if _is_rect_frame(coords):
                frames.append({"color": color, "bbox": _bbox(coords)})
        if frames:
            pairs[color] = frames
    return pairs


def _frame_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = bbox
    return (0.5 * (r0 + r1), 0.5 * (c0 + c1))


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _mark_reserved(occupied: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> None:
    h, w = occupied.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _is_free(occupied: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> bool:
    h, w = occupied.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not np.any(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1])


def _random_pattern(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    ih, iw = shape
    area = ih * iw
    for _ in range(PATTERN_ATTEMPTS):
        density = float(rng.uniform(0.22, 0.52))
        mask = rng.random(shape) < density
        count = int(mask.sum())
        if count < 2 or count > area - 2:
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        return mask
    raise RuntimeError("failed to sample a non-symmetric pattern")


def _sample_pair_layout(
    rng: np.random.Generator,
    occupied: np.ndarray,
    branch: str,
    color: int,
    size_cap: int,
) -> dict[str, object] | None:
    h, w = occupied.shape
    for _ in range(PAIR_PLACEMENT_ATTEMPTS):
        ih = int(rng.integers(2, size_cap + 1))
        iw = int(rng.integers(2, size_cap + 1))
        fh, fw = ih + 2, iw + 2
        gap = int(rng.integers(2, 7))
        if branch == "h":
            max_drift = min(4, fw + gap - 1)
            drift = int(rng.integers(-max_drift, max_drift + 1))
            pair_h = fh + abs(drift)
            pair_w = 2 * fw + gap
            if pair_h + 2 > h or pair_w + 2 > w:
                continue
            top = int(rng.integers(1, h - pair_h))
            left = int(rng.integers(1, w - pair_w))
            left_r = top + max(0, -drift)
            right_r = top + max(0, drift)
            box_a = (left_r, left, left_r + fh - 1, left + fw - 1)
            box_b = (right_r, left + fw + gap, right_r + fh - 1, left + 2 * fw + gap - 1)
        else:
            max_drift = min(4, fh + gap - 1)
            drift = int(rng.integers(-max_drift, max_drift + 1))
            pair_h = 2 * fh + gap
            pair_w = fw + abs(drift)
            if pair_h + 2 > h or pair_w + 2 > w:
                continue
            top = int(rng.integers(1, h - pair_h))
            left = int(rng.integers(1, w - pair_w))
            top_c = left + max(0, -drift)
            bottom_c = left + max(0, drift)
            box_a = (top, top_c, top + fh - 1, top_c + fw - 1)
            box_b = (top + fh + gap, bottom_c, top + 2 * fh + gap - 1, bottom_c + fw - 1)
        pair_bbox = (
            min(box_a[0], box_b[0]),
            min(box_a[1], box_b[1]),
            max(box_a[2], box_b[2]),
            max(box_a[3], box_b[3]),
        )
        if not _is_free(occupied, pair_bbox, pad=1):
            continue
        source_idx = int(rng.integers(0, 2))
        pattern = _random_pattern(rng, (ih, iw))
        return {
            "branch": branch,
            "color": color,
            "frames": [box_a, box_b],
            "source_idx": source_idx,
            "pattern": pattern,
            "pair_bbox": pair_bbox,
        }
    return None


def _apply_spec_output(grid: np.ndarray, spec: dict[str, object]) -> None:
    frames = spec["frames"]
    assert isinstance(frames, list)
    source_idx = int(spec["source_idx"])
    src = frames[source_idx]
    dst = frames[1 - source_idx]
    pattern = np.asarray(spec["pattern"], dtype=bool)
    transformed = np.fliplr(pattern) if spec["branch"] == "h" else np.flipud(pattern)
    dst_r0, dst_c0, dst_r1, dst_c1 = dst
    grid[dst_r0 + 1 : dst_r1, dst_c0 + 1 : dst_c1] = BG
    grid[dst_r0 + 1 : dst_r1, dst_c0 + 1 : dst_c1][transformed] = GREY


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each grey interior pattern into its same-color partner frame using a position-dependent flip.

    Input structure:
    - The background is black.
    - Colored pixels form several one-pixel-thick rectangular frames; each frame
      color appears exactly twice, creating an unambiguous pair.
    - Within each same-color pair, one frame interior is empty black and the
      other contains a grey non-symmetric pattern. Paired interiors have the
      same size.

    Rule:
    - Match the two frames of each color.
    - Keep the patterned frame unchanged and copy its grey interior pattern into
      the empty partner interior.
    - If the destination frame is primarily left/right of the source frame
      (horizontal center distance exceeds vertical center distance), apply a
      horizontal flip before copying; otherwise apply a vertical flip.
    - Leave all colored borders unchanged.

    Key insight:
    - Border color identifies the source-target pairing, while the relative
      placement of the paired frames decides whether the transferred pattern is
      mirrored left-right or up-down.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    frame_pairs = _find_frame_pairs(g, bg)
    assert frame_pairs, "expected at least one frame pair"

    out = g.copy()
    for color, frames in frame_pairs.items():
        assert len(frames) == 2, f"expected exactly two frames for color {color}"
        source = None
        target = None
        pattern_color = None
        for frame in frames:
            r0, c0, r1, c1 = frame["bbox"]  # type: ignore[index]
            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            inner_colors = np.unique(inner[inner != bg])
            if inner_colors.size == 0:
                assert target is None, "expected one empty frame per pair"
                target = frame
            else:
                assert source is None, "expected one patterned frame per pair"
                assert inner_colors.size == 1, "pattern should use a single visible color"
                source = frame
                pattern_color = int(inner_colors[0])
        assert source is not None and target is not None and pattern_color is not None

        src_r0, src_c0, src_r1, src_c1 = source["bbox"]  # type: ignore[index]
        dst_r0, dst_c0, dst_r1, dst_c1 = target["bbox"]  # type: ignore[index]
        src_inner = g[src_r0 + 1 : src_r1, src_c0 + 1 : src_c1]
        dst_inner_shape = (dst_r1 - dst_r0 - 1, dst_c1 - dst_c0 - 1)
        assert src_inner.shape == dst_inner_shape, "paired frame interiors must match"

        src_center = _frame_center(source["bbox"])  # type: ignore[arg-type,index]
        dst_center = _frame_center(target["bbox"])  # type: ignore[arg-type,index]
        dr = dst_center[0] - src_center[0]
        dc = dst_center[1] - src_center[1]
        pattern = src_inner != bg
        transformed = np.fliplr(pattern) if abs(dc) > abs(dr) else np.flipud(pattern)

        out[dst_r0 + 1 : dst_r1, dst_c0 + 1 : dst_c1] = bg
        out[dst_r0 + 1 : dst_r1, dst_c0 + 1 : dst_c1][transformed] = pattern_color
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        pair_count = int(rng.choice([2, 3]))
        min_side = 18 if pair_count == 2 else 22
        h = int(rng.integers(min_side, 31))
        w = int(rng.integers(min_side, 31))
        grid = np.full((h, w), BG, dtype=int)
        occupied = np.zeros((h, w), dtype=bool)
        size_cap = 6 if min(h, w) >= 24 else 5

        branches = ["h", "v"]
        if pair_count == 3:
            branches.append(str(rng.choice(["h", "v"])))
        rng.shuffle(branches)
        colors = [int(c) for c in rng.choice(FRAME_COLORS, size=pair_count, replace=False)]

        specs: list[dict[str, object]] = []
        ok = True
        for branch, color in zip(branches, colors):
            spec = _sample_pair_layout(rng, occupied, branch, color, size_cap)
            if spec is None:
                ok = False
                break
            specs.append(spec)
            _mark_reserved(occupied, spec["pair_bbox"], pad=1)  # type: ignore[arg-type]
        if not ok:
            continue

        for spec in specs:
            frames = spec["frames"]
            assert isinstance(frames, list)
            color = int(spec["color"])
            for bbox in frames:
                _draw_frame(grid, bbox, color)
            source_idx = int(spec["source_idx"])
            src_r0, src_c0, src_r1, src_c1 = frames[source_idx]
            pattern = np.asarray(spec["pattern"], dtype=bool)
            grid[src_r0 + 1 : src_r1, src_c0 + 1 : src_c1][pattern] = GREY

        expected = grid.copy()
        for spec in specs:
            _apply_spec_output(expected, spec)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            continue
        return grid
    raise RuntimeError("failed to generate a valid ARC input")
