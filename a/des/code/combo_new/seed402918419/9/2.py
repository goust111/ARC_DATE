from __future__ import annotations

from collections import deque
import random

import numpy as np


_NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Generator palette choices: fixed visible roles for clarity.
_BG = 8
_FILL = 5
_GEN_RETRIES = 500


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 4-connected components of a boolean mask."""
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in _NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_coords(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = _bbox(coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _mask_from_coords(coords: tuple[tuple[int, int], ...]) -> np.ndarray:
    h = max(r for r, _ in coords) + 1
    w = max(c for _, c in coords) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    return mask


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _variants(mask: np.ndarray) -> list[np.ndarray]:
    seen: set[tuple[bytes, tuple[int, int]]] = set()
    out: list[np.ndarray] = []
    for k in range(4):
        rot = np.rot90(mask, k)
        for cand in (rot, np.fliplr(rot)):
            tight = _tight_crop(cand)
            key = (tight.tobytes(), tuple(map(int, tight.shape)))
            if key in seen:
                continue
            seen.add(key)
            out.append(tight)
    return out


def _frame_info(grid: np.ndarray, coords: list[tuple[int, int]], bg: int) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = map(int, patch.shape)
    if h < 5 or w < 5:
        return None

    border_vals = set(int(v) for v in patch[0, :])
    border_vals.update(int(v) for v in patch[-1, :])
    border_vals.update(int(v) for v in patch[1:-1, 0])
    border_vals.update(int(v) for v in patch[1:-1, -1])
    if bg in border_vals or len(border_vals) != 1:
        return None
    border_color = int(next(iter(border_vals)))

    inner_vals = set(int(v) for v in np.unique(patch[1:-1, 1:-1]) if int(v) != bg)
    if len(inner_vals) != 1:
        return None
    fill_color = int(next(iter(inner_vals)))
    if fill_color == border_color:
        return None

    comp_colors = {int(grid[r, c]) for r, c in coords}
    if comp_colors != {border_color, fill_color}:
        return None

    holes = []
    for comp in _components(patch == bg):
        if all(0 < rr < h - 1 and 0 < cc < w - 1 for rr, cc in comp):
            holes.append(comp)
    if len(holes) != 1:
        return None

    hole_rel = holes[0]
    hole_abs = [(int(r0 + rr), int(c0 + cc)) for rr, cc in hole_rel]
    return {
        "border_color": border_color,
        "fill_color": fill_color,
        "hole_shape": _normalize_coords(hole_rel),
        "hole_abs": hole_abs,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each framed cavity with the matching candidate piece color and erase the candidate row.

    Input structure:
    - The background is the most common color.
    - Several large 4-connected two-color rectangular regions appear in the upper area: each has a 1-cell border of one color and a uniform interior fill of another color.
    - Inside each large region, one connected background-colored hole forms a missing piece shape; elsewhere, separate small monochrome candidate shapes sit on the background.

    Rule:
    - Detect each large framed region and extract its internal background hole as a 4-connected shape, normalized by translation.
    - Detect each separate candidate piece and extract its normalized 4-connected shape.
    - Match holes to candidate pieces by exact shape equality under translation only, then recolor the hole cells with the matched candidate's color.
    - Remove all candidate pieces by painting their cells back to the background; keep every frame border and non-hole interior cell unchanged.

    Key insight:
    - The missing object is encoded as negative space: convert each cavity into a shape signature, then use that signature to transfer the candidate piece's color into the framed hole.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    out = g.copy()

    components = _components(g != bg)
    frames: list[dict] = []
    candidates: list[dict] = []

    for coords in components:
        colors = {int(g[r, c]) for r, c in coords}
        if len(colors) == 2:
            info = _frame_info(g, coords, bg)
            if info is not None:
                frames.append(info)
            continue
        if len(colors) == 1:
            candidates.append(
                {
                    "color": int(next(iter(colors))),
                    "shape": _normalize_coords(coords),
                    "coords": [(int(r), int(c)) for r, c in coords],
                }
            )

    assert frames, "expected at least one framed region"
    assert len(frames) == len(candidates), "expected one candidate per frame"

    by_shape: dict[tuple[tuple[int, int], ...], int] = {}
    for cand in candidates:
        shape = cand["shape"]
        assert shape not in by_shape, "candidate shapes must be unique"
        by_shape[shape] = int(cand["color"])

    for frame in frames:
        color = by_shape[frame["hole_shape"]]
        for r, c in frame["hole_abs"]:
            out[int(r), int(c)] = color

    for cand in candidates:
        for r, c in cand["coords"]:
            out[int(r), int(c)] = bg

    return out


_BASE_SHAPES = [
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1], [1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 0], [0, 1, 0]], dtype=bool),
]


def _random_shape_set(count: int) -> list[np.ndarray] | None:
    """Pick unique translated shapes after random orientation changes."""
    order = list(range(len(_BASE_SHAPES)))
    random.shuffle(order)
    chosen: list[np.ndarray] = []
    seen: set[tuple[bytes, tuple[int, int]]] = set()
    for idx in order:
        mask = random.choice(_variants(_BASE_SHAPES[idx]))
        key = (mask.tobytes(), tuple(map(int, mask.shape)))
        if key in seen:
            continue
        seen.add(key)
        chosen.append(mask)
        if len(chosen) == count:
            return chosen
    return None


def _make_candidate_patch(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = int(color)
    return patch


def _make_frame_patch(mask: np.ndarray, border_color: int, fill_color: int, bg: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    sh, sw = map(int, mask.shape)
    extra_h = random.randint(0, 2)
    extra_w = random.randint(0, 2)
    h = sh + 4 + extra_h
    w = sw + 4 + extra_w
    patch = np.full((h, w), int(fill_color), dtype=int)
    patch[0, :] = int(border_color)
    patch[-1, :] = int(border_color)
    patch[:, 0] = int(border_color)
    patch[:, -1] = int(border_color)

    hole_r0 = random.randint(2, h - sh - 2)
    hole_c0 = random.randint(2, w - sw - 2)
    hole_rel: list[tuple[int, int]] = []
    for rr, cc in np.argwhere(mask):
        rr = int(rr)
        cc = int(cc)
        patch[hole_r0 + rr, hole_c0 + cc] = int(bg)
        hole_rel.append((int(hole_r0 + rr), int(hole_c0 + cc)))
    return patch, hole_rel


def _row_layout(sizes: list[tuple[int, int]], indices: list[int], gap_range: tuple[int, int]) -> tuple[dict[int, tuple[int, int]], int, int]:
    row_h = max(sizes[idx][0] for idx in indices)
    pos: dict[int, tuple[int, int]] = {}
    x = 0
    for j, idx in enumerate(indices):
        if j > 0:
            x += random.randint(*gap_range)
        h, w = sizes[idx]
        y = random.randint(0, row_h - h)
        pos[idx] = (int(y), int(x))
        x += int(w)
    return pos, int(row_h), int(x)


def _top_layout(sizes: list[tuple[int, int]]) -> tuple[dict[int, tuple[int, int]], int, int]:
    count = len(sizes)
    order = list(range(count))
    random.shuffle(order)

    if count == 4:
        split = random.choice([1, 2, 3])
        rows = [order[:split], order[split:]]
    else:
        if random.random() < 0.5:
            rows = [order]
        else:
            split = random.choice([1, 2])
            rows = [order[:split], order[split:]]

    positions: dict[int, tuple[int, int]] = {}
    y = 0
    max_w = 0
    for ridx, row in enumerate(rows):
        row_pos, row_h, row_w = _row_layout(sizes, row, (1, 3))
        for idx, (rr, cc) in row_pos.items():
            positions[idx] = (int(y + rr), int(cc))
        max_w = max(max_w, int(row_w))
        y += int(row_h)
        if ridx != len(rows) - 1:
            y += random.randint(1, 2)
    return positions, int(y), int(max_w)


def _candidate_layout(sizes: list[tuple[int, int]]) -> tuple[dict[int, tuple[int, int]], int, int]:
    order = list(range(len(sizes)))
    random.shuffle(order)
    return _row_layout(sizes, order, (1, 3))


def _paste(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    sub = canvas[r0 : r0 + h, c0 : c0 + w]
    mask = patch != bg
    assert not np.any(sub[mask] != bg), "patch overlap"
    sub[mask] = patch[mask]


def _build_output_shape(frame_infos: list[dict], frame_positions: dict[int, tuple[int, int]], shape_to_piece_color: dict[tuple[tuple[int, int], ...], int], final_shape: tuple[int, int]) -> np.ndarray:
    out = np.full(final_shape, _BG, dtype=int)
    for idx, info in enumerate(frame_infos):
        patch = info["patch"]
        r0, c0 = frame_positions[idx]
        _paste(out, patch, r0, c0, _BG)
        fill_color = shape_to_piece_color[info["shape_key"]]
        for rr, cc in info["hole_rel"]:
            out[r0 + rr, c0 + cc] = int(fill_color)
    return out


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(3, 4)
    shapes = _random_shape_set(count)
    if shapes is None:
        return None

    palette = [c for c in range(10) if c not in {_BG, _FILL}]
    random.shuffle(palette)
    border_colors = palette[:count]
    piece_colors = palette[count : 2 * count]

    frame_infos: list[dict] = []
    candidate_patches: list[np.ndarray] = []
    shape_to_piece_color: dict[tuple[tuple[int, int], ...], int] = {}

    for idx in range(count):
        shape_mask = np.asarray(shapes[idx], dtype=bool)
        shape_key = _normalize_coords([(int(r), int(c)) for r, c in np.argwhere(shape_mask)])
        if shape_key in shape_to_piece_color:
            return None
        shape_to_piece_color[shape_key] = int(piece_colors[idx])

        frame_patch, hole_rel = _make_frame_patch(shape_mask, border_colors[idx], _FILL, _BG)
        frame_infos.append(
            {
                "patch": frame_patch,
                "shape_key": shape_key,
                "hole_rel": hole_rel,
            }
        )
        candidate_patches.append(_make_candidate_patch(shape_mask, piece_colors[idx], _BG))

    frame_sizes = [tuple(map(int, info["patch"].shape)) for info in frame_infos]
    candidate_sizes = [tuple(map(int, patch.shape)) for patch in candidate_patches]

    frame_positions, top_h, top_w = _top_layout(frame_sizes)
    candidate_positions, cand_h, cand_w = _candidate_layout(candidate_sizes)
    divider_gap = random.randint(1, 3)
    content_h = top_h + divider_gap + cand_h
    content_w = max(top_w, cand_w)
    if content_h > 30 or content_w > 30:
        return None

    final_h = content_h + random.randint(0, min(4, 30 - content_h))
    final_w = content_w + random.randint(0, min(4, 30 - content_w))
    top_offset = random.randint(0, final_h - content_h)
    frame_x_offset = random.randint(0, final_w - top_w)
    cand_x_offset = random.randint(0, final_w - cand_w)

    input_grid = np.full((final_h, final_w), _BG, dtype=int)
    output_grid = _build_output_shape(
        frame_infos,
        {idx: (top_offset + r, frame_x_offset + c) for idx, (r, c) in frame_positions.items()},
        shape_to_piece_color,
        (final_h, final_w),
    )

    for idx, info in enumerate(frame_infos):
        r0, c0 = frame_positions[idx]
        _paste(input_grid, info["patch"], top_offset + r0, frame_x_offset + c0, _BG)

    cand_top = top_offset + top_h + divider_gap
    for idx, patch in enumerate(candidate_patches):
        r0, c0 = candidate_positions[idx]
        _paste(input_grid, patch, cand_top + r0, cand_x_offset + c0, _BG)

    if _most_common_color(input_grid) != _BG:
        return None
    if np.array_equal(main(input_grid), output_grid):
        return input_grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
