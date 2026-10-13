from __future__ import annotations

from collections import Counter

import numpy as np


SIDES = ("top", "bottom", "left", "right")
MAX_GENERATION_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(coords)
    return parts


def _tight_frame_mask(height: int, width: int, missing: str) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if missing != "top":
        mask[0, :] = True
    if missing != "bottom":
        mask[-1, :] = True
    if missing != "left":
        mask[:, 0] = True
    if missing != "right":
        mask[:, -1] = True
    return mask


def _full_frame_mask(height: int, width: int, missing: str) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True

    if missing == "top":
        mask[0, :] = False
    elif missing == "bottom":
        mask[-1, :] = False
    elif missing == "left":
        mask[:, 0] = False
    elif missing == "right":
        mask[:, -1] = False
    else:
        raise ValueError(f"unknown side: {missing}")
    return mask


def _classify_frame(coords: list[tuple[int, int]]) -> dict[str, int | str] | None:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None

    local = np.zeros((height, width), dtype=bool)
    for r, c in coords:
        local[r - r0, c - c0] = True

    if np.any(local[1:-1, 1:-1]):
        return None

    for missing in SIDES:
        if not np.array_equal(local, _tight_frame_mask(height, width, missing)):
            continue

        if missing == "top":
            full_r0, full_c0, full_r1, full_c1 = r0 - 1, c0, r1, c1
        elif missing == "bottom":
            full_r0, full_c0, full_r1, full_c1 = r0, c0, r1 + 1, c1
        elif missing == "left":
            full_r0, full_c0, full_r1, full_c1 = r0, c0 - 1, r1, c1
        elif missing == "right":
            full_r0, full_c0, full_r1, full_c1 = r0, c0, r1, c1 + 1
        else:
            raise ValueError(f"unknown side: {missing}")

        if full_r0 < 0 or full_c0 < 0:
            return None
        return {
            "r0": int(full_r0),
            "c0": int(full_c0),
            "r1": int(full_r1),
            "c1": int(full_c1),
            "missing": missing,
        }
    return None


def _extract_scene(grid: np.ndarray) -> tuple[int, list[dict[str, int | str]], list[dict[str, int]]]:
    bg = _background_color(grid)
    frames: list[dict[str, int | str]] = []
    markers: list[dict[str, int]] = []

    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for coords in _components(grid == color):
            if len(coords) == 1:
                r, c = coords[0]
                markers.append({"row": int(r), "col": int(c), "color": int(color)})
                continue

            frame = _classify_frame(coords)
            if frame is not None:
                frame["color"] = int(color)
                frames.append(frame)

    return bg, frames, markers


def _marker_distance(marker: dict[str, int], frame: dict[str, int | str]) -> int | None:
    row = marker["row"]
    col = marker["col"]
    r0 = int(frame["r0"])
    c0 = int(frame["c0"])
    r1 = int(frame["r1"])
    c1 = int(frame["c1"])
    missing = str(frame["missing"])

    if missing == "top" and c0 <= col <= c1 and row < r0:
        return r0 - row
    if missing == "bottom" and c0 <= col <= c1 and row > r1:
        return row - r1
    if missing == "left" and r0 <= row <= r1 and col < c0:
        return c0 - col
    if missing == "right" and r0 <= row <= r1 and col > c1:
        return col - c1
    return None


def _resolve_fill_color(markers: list[dict[str, int]]) -> int:
    counts = Counter(marker["color"] for marker in markers)
    best_count = max(counts.values())
    candidates = [color for color, count in counts.items() if count == best_count]
    if len(candidates) == 1:
        return int(candidates[0])

    # Valid samples avoid ties. If one appears, prefer the tied color with the
    # closest supporting marker so the decision still depends on geometry.
    best_by_distance = min(
        min(marker["distance"] for marker in markers if marker["color"] == color)
        for color in candidates
    )
    for marker in markers:
        if marker["color"] in candidates and marker["distance"] == best_by_distance:
            return int(marker["color"])
    raise AssertionError("tie resolution failed")


def _paint_missing_side(grid: np.ndarray, frame: dict[str, int | str], color: int) -> None:
    r0 = int(frame["r0"])
    c0 = int(frame["c0"])
    r1 = int(frame["r1"])
    c1 = int(frame["c1"])
    missing = str(frame["missing"])

    if missing == "top":
        grid[r0, c0 : c1 + 1] = color
    elif missing == "bottom":
        grid[r1, c0 : c1 + 1] = color
    elif missing == "left":
        grid[r0 : r1 + 1, c0] = color
    elif missing == "right":
        grid[r0 : r1 + 1, c1] = color
    else:
        raise ValueError(f"unknown side: {missing}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align scattered markers to the nearest incomplete rectangular frame and use
    the dominant marker color to complete the missing side.
    """
    g = np.asarray(grid, dtype=int)
    bg, frames, markers = _extract_scene(g)
    out = g.copy()

    assignments: dict[int, list[dict[str, int]]] = {idx: [] for idx in range(len(frames))}
    for marker in markers:
        candidates: list[tuple[int, int]] = []
        for frame_idx, frame in enumerate(frames):
            distance = _marker_distance(marker, frame)
            if distance is not None:
                candidates.append((int(distance), int(frame_idx)))
        if not candidates:
            continue

        best_distance = min(distance for distance, _ in candidates)
        best_frames = [frame_idx for distance, frame_idx in candidates if distance == best_distance]
        if len(best_frames) != 1:
            continue

        assigned = dict(marker)
        assigned["distance"] = int(best_distance)
        assignments[best_frames[0]].append(assigned)

    for frame_idx, frame_markers in assignments.items():
        if not frame_markers:
            continue
        fill_color = _resolve_fill_color(frame_markers)
        for marker in frame_markers:
            out[marker["row"], marker["col"]] = bg
        _paint_missing_side(out, frames[frame_idx], fill_color)

    return out


def _paint_broken_frame(
    grid: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    color: int,
    missing: str,
) -> None:
    mask = _full_frame_mask(height, width, missing)
    patch = grid[top : top + height, left : left + width]
    patch[mask] = color


def _sample_layout(rng: np.random.Generator) -> tuple[int, int, int, np.ndarray, np.ndarray] | None:
    n_frames = int(rng.integers(2, 5))

    frame_max = {2: 7, 3: 6, 4: 5}[n_frames]
    frame_h = int(rng.integers(4, frame_max + 1))
    frame_w = int(rng.integers(4, frame_max + 1))

    gap_high = 5 if n_frames <= 3 else 4
    row_gaps = rng.integers(2, gap_high, size=n_frames + 1)
    col_gaps = rng.integers(2, gap_high, size=n_frames + 1)

    height = int(n_frames * frame_h + int(row_gaps.sum()))
    width = int(n_frames * frame_w + int(col_gaps.sum()))
    if not (12 <= height <= 30 and 12 <= width <= 30):
        return None
    return n_frames, frame_h, frame_w, row_gaps.astype(int), col_gaps.astype(int)


def _starts(frame_len: int, gaps: np.ndarray) -> list[int]:
    starts: list[int] = []
    cursor = 0
    for idx in range(len(gaps) - 1):
        cursor += int(gaps[idx])
        starts.append(cursor)
        cursor += int(frame_len)
    return starts


def _sample_marker_colors(
    rng: np.random.Generator, count: int, fill_color: int, bg: int
) -> list[int]:
    min_majority = count // 2 + 1
    majority = int(rng.integers(min_majority, count + 1))
    colors = [int(fill_color)] * majority

    if majority < count:
        noise_pool = np.array([c for c in range(10) if c not in {bg, fill_color}], dtype=int)
        noise = rng.choice(noise_pool, size=count - majority, replace=True)
        colors.extend(int(value) for value in noise.tolist())

    rng.shuffle(colors)
    return colors


def _marker_positions(
    rng: np.random.Generator,
    top: int,
    left: int,
    height: int,
    width: int,
    missing: str,
    row_slot: int,
    col_slot: int,
    row_gaps: np.ndarray,
    col_gaps: np.ndarray,
) -> list[tuple[int, int]]:
    lane_len = width if missing in {"top", "bottom"} else height
    count = int(rng.integers(2, min(4, lane_len) + 1))

    if missing == "top":
        cols = rng.choice(np.arange(left, left + width), size=count, replace=False)
        dists = rng.integers(1, int(row_gaps[row_slot]) + 1, size=count)
        return [(top - int(dist), int(col)) for dist, col in zip(dists, cols)]
    if missing == "bottom":
        cols = rng.choice(np.arange(left, left + width), size=count, replace=False)
        dists = rng.integers(1, int(row_gaps[row_slot + 1]) + 1, size=count)
        return [(top + height - 1 + int(dist), int(col)) for dist, col in zip(dists, cols)]
    if missing == "left":
        rows = rng.choice(np.arange(top, top + height), size=count, replace=False)
        dists = rng.integers(1, int(col_gaps[col_slot]) + 1, size=count)
        return [(int(row), left - int(dist)) for row, dist in zip(rows, dists)]
    if missing == "right":
        rows = rng.choice(np.arange(top, top + height), size=count, replace=False)
        dists = rng.integers(1, int(col_gaps[col_slot + 1]) + 1, size=count)
        return [(int(row), left + width - 1 + int(dist)) for row, dist in zip(rows, dists)]
    raise ValueError(f"unknown side: {missing}")


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    layout = _sample_layout(rng)
    if layout is None:
        return None
    n_frames, frame_h, frame_w, row_gaps, col_gaps = layout

    bg = int(rng.integers(0, 10))
    palette = [color for color in range(10) if color != bg]
    palette = [int(color) for color in rng.permutation(np.array(palette, dtype=int)).tolist()]
    frame_colors = palette[:n_frames]
    fill_colors = palette[n_frames : 2 * n_frames]

    row_starts = _starts(frame_h, row_gaps)
    col_starts = _starts(frame_w, col_gaps)
    col_order = rng.permutation(n_frames).astype(int)

    height = int(n_frames * frame_h + int(row_gaps.sum()))
    width = int(n_frames * frame_w + int(col_gaps.sum()))
    inp = np.full((height, width), bg, dtype=int)
    out = np.full((height, width), bg, dtype=int)

    sampled_sides = [SIDES[int(idx)] for idx in rng.integers(0, 4, size=n_frames)]
    if len(set(sampled_sides)) == 1:
        sampled_sides[-1] = SIDES[(SIDES.index(sampled_sides[-1]) + 1) % 4]

    for idx in range(n_frames):
        top = int(row_starts[idx])
        left = int(col_starts[int(col_order[idx])])
        border_color = int(frame_colors[idx])
        fill_color = int(fill_colors[idx])
        missing = sampled_sides[idx]

        _paint_broken_frame(inp, top, left, frame_h, frame_w, border_color, missing)
        _paint_broken_frame(out, top, left, frame_h, frame_w, border_color, missing)
        _paint_missing_side(
            out,
            {"r0": top, "c0": left, "r1": top + frame_h - 1, "c1": left + frame_w - 1, "missing": missing},
            fill_color,
        )

        coords = _marker_positions(
            rng,
            top,
            left,
            frame_h,
            frame_w,
            missing,
            idx,
            int(col_order[idx]),
            row_gaps,
            col_gaps,
        )
        marker_colors = _sample_marker_colors(rng, len(coords), fill_color, bg)
        for (row, col), color in zip(coords, marker_colors):
            inp[row, col] = int(color)

    if _background_color(inp) != bg:
        return None
    if np.count_nonzero(inp == bg) <= inp.size // 2:
        return None

    solved = main(inp)
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
