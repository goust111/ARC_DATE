from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    cols = set(int(x) for x in np.unique(grid))
    if exclude is not None:
        cols -= set(int(x) for x in exclude)
    return sorted(cols)


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(int(x) for x in grid.ravel()).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:, :] = patch
        return
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _neighbors(r: int, c: int, shape: tuple[int, int], connectivity: int) -> list[Coord]:
    h, w = shape
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    out = []
    for dr, dc in offsets:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def components_mask(mask: np.ndarray, *, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, mask.shape, connectivity):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _full_rows(grid: np.ndarray, color: int) -> list[int]:
    return [r for r in range(grid.shape[0]) if np.all(grid[r, :] == int(color))]


def _full_cols(grid: np.ndarray, color: int) -> list[int]:
    return [c for c in range(grid.shape[1]) if np.all(grid[:, c] == int(color))]


def _is_regular(lines: list[int]) -> bool:
    if len(lines) < 2:
        return False
    diffs = np.diff(lines)
    return bool(np.all(diffs == diffs[0]) and diffs[0] > 1)


def _detect_frame_lattice(grid: np.ndarray) -> tuple[int, str, list[int], list[int]]:
    h, w = grid.shape
    candidates: list[tuple[int, str, list[int], list[int]]] = []
    for color in colors_present(grid):
        rows = _full_rows(grid, color)
        cols = _full_cols(grid, color)
        if len(rows) == 2 and rows[0] == 0 and rows[1] == h - 1 and len(cols) >= 4 and cols[0] == 0 and cols[-1] == w - 1 and _is_regular(cols):
            candidates.append((int(color), "horizontal", rows, cols))
        if len(cols) == 2 and cols[0] == 0 and cols[1] == w - 1 and len(rows) >= 4 and rows[0] == 0 and rows[-1] == h - 1 and _is_regular(rows):
            candidates.append((int(color), "vertical", rows, cols))
    assert len(candidates) == 1, f"expected one separator lattice, got {candidates}"
    return candidates[0]


def _cell_slices(orientation: str, rows: list[int], cols: list[int]) -> list[tuple[slice, slice]]:
    if orientation == "horizontal":
        return [(slice(rows[0] + 1, rows[1]), slice(cols[i] + 1, cols[i + 1])) for i in range(len(cols) - 1)]
    return [(slice(rows[i] + 1, rows[i + 1]), slice(cols[0] + 1, cols[1])) for i in range(len(rows) - 1)]


def _sprite_patch_from_coords(cell: np.ndarray, coords: np.ndarray, color: int) -> tuple[np.ndarray, BBox]:
    box = bbox_coords(coords)
    r0, c0, r1, c1 = box
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), -1, dtype=int)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = int(color)
    return patch, box


def _chebyshev_disconnected(point: Coord, coords: np.ndarray) -> bool:
    rr, cc = point
    return all(max(abs(rr - int(r)), abs(cc - int(c))) > 1 for r, c in coords)


def _parse_frame(cell: np.ndarray, bg: int) -> dict:
    if np.all(cell == int(bg)):
        return {"empty": True}

    comps = []
    for color in colors_present(cell, exclude={bg}):
        for coords in components_mask(cell == int(color), connectivity=4):
            comps.append({"color": int(color), "coords": coords, "area": int(coords.shape[0])})

    assert len(comps) in {1, 2}, f"unexpected frame components: {[(c['color'], c['area']) for c in comps]}"
    sprite_candidates = [comp for comp in comps if comp["area"] > 1]
    assert len(sprite_candidates) == 1, "expected one multi-pixel sprite"
    sprite = sprite_candidates[0]
    sprite_patch, sprite_box = _sprite_patch_from_coords(cell, sprite["coords"], sprite["color"])

    indicator: Coord | None = None
    indicator_color: int | None = None
    indicator_candidates = [comp for comp in comps if comp["area"] == 1]
    if indicator_candidates:
        assert len(indicator_candidates) == 1, "expected at most one indicator pixel"
        indicator_comp = indicator_candidates[0]
        indicator = tuple(int(x) for x in indicator_comp["coords"][0])
        indicator_color = int(indicator_comp["color"])
        assert _chebyshev_disconnected(indicator, sprite["coords"]), "indicator must be isolated from sprite"

    return {
        "empty": False,
        "sprite_color": int(sprite["color"]),
        "sprite_patch": sprite_patch,
        "sprite_bbox": sprite_box,
        "top_left": (int(sprite_box[0]), int(sprite_box[1])),
        "indicator": indicator,
        "indicator_color": indicator_color,
    }


def _build_missing_cell(
    cell_shape: tuple[int, int],
    bg: int,
    sprite_patch: np.ndarray,
    sprite_top_left: Coord,
    indicator: Coord | None,
    indicator_color: int | None,
) -> np.ndarray:
    cell = np.full(cell_shape, int(bg), dtype=int)
    paste(cell, sprite_patch, sprite_top_left[0], sprite_top_left[1], transparent=-1)
    if indicator is not None:
        assert indicator_color is not None
        cell[indicator] = int(indicator_color)
    return cell


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the one empty frame in a separator-divided frame sequence using previous-frame motion and next-frame back-inference.

    Input structure:
    - A single row or column of equal cells is separated by full same-color lines with an outer border.
    - Exactly one cell is completely empty; every other cell contains the same monochrome sprite.
    - Every non-final visible frame also has one isolated indicator pixel of another shared color, while the last frame has no indicator.

    Rule:
    - Within each frame, indicator position minus the sprite bbox's top-left position equals the sprite-top-left displacement to the next frame.
    - Recover the missing frame's sprite by applying the previous frame's indicator vector to the previous sprite position.
    - Recover the missing frame's own indicator from the displacement to the following sprite; if the missing frame is the last frame, draw no indicator.
    - Leave all separators and all already-complete frames unchanged.

    Key insight:
    - The indicator is a per-frame motion code: the empty frame's sprite comes only from the previous frame, while its indicator comes only from the following frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    _sep_color, orientation, rows, cols = _detect_frame_lattice(g)
    slices = _cell_slices(orientation, rows, cols)
    cells = [g[rs, cs].copy() for rs, cs in slices]
    empty_cells = [i for i, cell in enumerate(cells) if len(colors_present(cell)) == 1]
    assert len(empty_cells) == 1, "expected exactly one empty cell"
    missing_idx = empty_cells[0]
    assert missing_idx > 0, "missing frame must not be first"

    bg = int(cells[missing_idx][0, 0])
    parsed = [_parse_frame(cell, bg) for cell in cells]

    non_empty = [frame for frame in parsed if not frame["empty"]]
    sprite_colors = {frame["sprite_color"] for frame in non_empty}
    assert len(sprite_colors) == 1, "expected a shared sprite color"
    sprite_templates = [frame["sprite_patch"] for frame in non_empty]
    template = sprite_templates[0]
    assert all(np.array_equal(patch, template) for patch in sprite_templates[1:]), "expected a shared sprite shape"

    indicator_colors = {frame["indicator_color"] for frame in non_empty if frame["indicator_color"] is not None}
    assert len(indicator_colors) == 1, "expected a shared indicator color"
    indicator_color = next(iter(indicator_colors), None)

    n_frames = len(parsed)
    for i, frame in enumerate(parsed):
        if frame["empty"]:
            continue
        if i == n_frames - 1:
            if missing_idx != n_frames - 1:
                assert frame["indicator"] is None, "the final visible frame must have no indicator"
        else:
            assert frame["indicator"] is not None, "every non-final visible frame must have an indicator"

    for i in range(n_frames - 1):
        cur = parsed[i]
        nxt = parsed[i + 1]
        if cur["empty"] or nxt["empty"]:
            continue
        dr = nxt["top_left"][0] - cur["top_left"][0]
        dc = nxt["top_left"][1] - cur["top_left"][1]
        vector = (cur["indicator"][0] - cur["top_left"][0], cur["indicator"][1] - cur["top_left"][1])
        assert vector == (dr, dc), "indicator must encode the next-frame displacement"

    prev = parsed[missing_idx - 1]
    assert not prev["empty"] and prev["indicator"] is not None
    missing_top_left = prev["indicator"]

    if missing_idx < n_frames - 1:
        nxt = parsed[missing_idx + 1]
        assert not nxt["empty"]
        missing_indicator = nxt["top_left"]
        next_vector = (missing_indicator[0] - missing_top_left[0], missing_indicator[1] - missing_top_left[1])
    else:
        missing_indicator = None
        next_vector = None

    cell_shape = cells[missing_idx].shape
    th, tw = template.shape
    mr, mc = missing_top_left
    assert 0 <= mr <= cell_shape[0] - th and 0 <= mc <= cell_shape[1] - tw, "missing sprite must fit inside its cell"
    if missing_indicator is not None:
        ir, ic = missing_indicator
        assert 0 <= ir < cell_shape[0] and 0 <= ic < cell_shape[1], "missing indicator must stay inside its cell"
        sprite_coords = np.argwhere(template != -1) + np.array([[mr, mc]])
        assert _chebyshev_disconnected(missing_indicator, sprite_coords), "reconstructed indicator must stay isolated"

    out = g.copy()
    filled = _build_missing_cell(
        cell_shape=cell_shape,
        bg=bg,
        sprite_patch=template,
        sprite_top_left=missing_top_left,
        indicator=missing_indicator,
        indicator_color=indicator_color,
    )
    rs, cs = slices[missing_idx]
    out[rs, cs] = filled

    full_positions = [frame["top_left"] if not frame["empty"] else missing_top_left for frame in parsed]
    full_vectors = []
    for i in range(n_frames - 1):
        dr = full_positions[i + 1][0] - full_positions[i][0]
        dc = full_positions[i + 1][1] - full_positions[i][1]
        full_vectors.append((dr, dc))
    assert len(set(full_vectors)) == len(full_vectors), "adjacent displacements must all differ"
    if next_vector is not None:
        assert missing_indicator == (
            missing_top_left[0] + next_vector[0],
            missing_top_left[1] + next_vector[1],
        )

    return out


N_FRAMES = 4
MAX_ATTEMPTS = 400


def _random_sprite(shape_h: int, shape_w: int) -> np.ndarray:
    for _ in range(200):
        target = random.randint(max(3, (shape_h * shape_w) // 2), max(3, shape_h * shape_w - 1))
        cells = {(0, 0)}
        while len(cells) < target:
            frontier = set()
            for r, c in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < shape_h and 0 <= nc < shape_w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(list(frontier)))
        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        if 0 not in rows or 0 not in cols or shape_h - 1 not in rows or shape_w - 1 not in cols:
            continue
        sprite = np.zeros((shape_h, shape_w), dtype=int)
        for r, c in cells:
            sprite[r, c] = 1
        return sprite
    raise RuntimeError("failed to sample sprite")


def _valid_transition(sprite_mask: np.ndarray, pos_a: Coord, pos_b: Coord) -> bool:
    ah, aw = sprite_mask.shape
    if pos_a[0] <= pos_b[0] < pos_a[0] + ah and pos_a[1] <= pos_b[1] < pos_a[1] + aw:
        return False
    sprite_coords = np.argwhere(sprite_mask)
    for r, c in sprite_coords:
        rr = pos_a[0] + int(r)
        cc = pos_a[1] + int(c)
        if max(abs(pos_b[0] - rr), abs(pos_b[1] - cc)) <= 1:
            return False
    return True


def _find_motion_path(cell_h: int, cell_w: int, sprite_mask: np.ndarray) -> list[Coord] | None:
    sh, sw = sprite_mask.shape
    positions = [(r, c) for r in range(cell_h - sh + 1) for c in range(cell_w - sw + 1)]
    random.shuffle(positions)

    def dfs(path: list[Coord], vectors: list[Coord]) -> list[Coord] | None:
        if len(path) == N_FRAMES:
            return list(path)
        prev = path[-1]
        candidates = positions[:]
        random.shuffle(candidates)
        for nxt in candidates:
            if nxt in path:
                continue
            vec = (nxt[0] - prev[0], nxt[1] - prev[1])
            if vec == (0, 0) or vec in vectors:
                continue
            if not _valid_transition(sprite_mask, prev, nxt):
                continue
            path.append(nxt)
            vectors.append(vec)
            solved = dfs(path, vectors)
            if solved is not None:
                return solved
            path.pop()
            vectors.pop()
        return None

    for start in positions:
        solved = dfs([start], [])
        if solved is not None:
            return solved
    return None


def _indicator_slots(cell_h: int, cell_w: int, sprite_mask: np.ndarray, top_left: Coord) -> list[Coord]:
    out = []
    for r in range(cell_h):
        for c in range(cell_w):
            if _valid_transition(sprite_mask, top_left, (r, c)):
                out.append((r, c))
    return out


def _top_left_candidates_from_next(cell_h: int, cell_w: int, sprite_mask: np.ndarray, next_top_left: Coord) -> list[Coord]:
    sh, sw = sprite_mask.shape
    out = []
    for r in range(cell_h - sh + 1):
        for c in range(cell_w - sw + 1):
            if _valid_transition(sprite_mask, (r, c), next_top_left):
                out.append((r, c))
    return out


def _render_sequence(
    *,
    orientation: str,
    bg: int,
    sep: int,
    sprite_color: int,
    indicator_color: int,
    sprite_mask: np.ndarray,
    positions: list[Coord],
    missing_idx: int,
    cell_h: int,
    cell_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    if orientation == "horizontal":
        grid = np.full((cell_h + 2, N_FRAMES * cell_w + N_FRAMES + 1), int(sep), dtype=int)
    else:
        grid = np.full((N_FRAMES * cell_h + N_FRAMES + 1, cell_w + 2), int(sep), dtype=int)

    out = grid.copy()
    inp = grid.copy()
    sprite_patch = np.where(sprite_mask, int(sprite_color), -1)

    if orientation == "horizontal":
        boxes = [(slice(1, 1 + cell_h), slice(1 + i * (cell_w + 1), 1 + i * (cell_w + 1) + cell_w)) for i in range(N_FRAMES)]
    else:
        boxes = [(slice(1 + i * (cell_h + 1), 1 + i * (cell_h + 1) + cell_h), slice(1, 1 + cell_w)) for i in range(N_FRAMES)]

    for i, (rs, cs) in enumerate(boxes):
        out[rs, cs] = int(bg)
        inp[rs, cs] = int(bg)
        cell = np.full((cell_h, cell_w), int(bg), dtype=int)
        paste(cell, sprite_patch, positions[i][0], positions[i][1], transparent=-1)
        if i < N_FRAMES - 1:
            cell[positions[i + 1]] = int(indicator_color)
        out[rs, cs] = cell
        if i != missing_idx:
            inp[rs, cs] = cell

    return inp, out


def _semantic_checks(
    *,
    cell_h: int,
    cell_w: int,
    sprite_mask: np.ndarray,
    positions: list[Coord],
    missing_idx: int,
) -> bool:
    vectors = [(positions[i + 1][0] - positions[i][0], positions[i + 1][1] - positions[i][1]) for i in range(N_FRAMES - 1)]
    if len(set(vectors)) != len(vectors):
        return False

    for i in range(N_FRAMES - 1):
        if not _valid_transition(sprite_mask, positions[i], positions[i + 1]):
            return False

    if missing_idx < N_FRAMES - 1:
        if positions[missing_idx - 1][0] + positions[missing_idx + 1][0] == 2 * positions[missing_idx][0] and positions[missing_idx - 1][1] + positions[missing_idx + 1][1] == 2 * positions[missing_idx][1]:
            return False
        if len(_top_left_candidates_from_next(cell_h, cell_w, sprite_mask, positions[missing_idx + 1])) <= 1:
            return False
        if len(_indicator_slots(cell_h, cell_w, sprite_mask, positions[missing_idx])) <= 1:
            return False
    else:
        if vectors[-1] == vectors[-2]:
            return False

    return True


def _sample_case() -> tuple[np.ndarray, np.ndarray] | None:
    orientation = random.choice(["horizontal", "vertical"])
    if orientation == "horizontal":
        cell_h = random.randint(5, 8)
        cell_w = random.randint(5, 6)
    else:
        cell_h = random.randint(5, 6)
        cell_w = random.randint(5, 8)

    palette = random.sample(range(10), 4)
    bg, sep, sprite_color, indicator_color = palette

    sprite_h = random.randint(2, min(4, cell_h - 2))
    sprite_w = random.randint(2, min(4, cell_w - 2))
    sprite_mask = _random_sprite(sprite_h, sprite_w)
    positions = _find_motion_path(cell_h, cell_w, sprite_mask)
    if positions is None:
        return None

    missing_idx = random.choices([1, 2, 3], weights=[3, 3, 2])[0]
    if not _semantic_checks(
        cell_h=cell_h,
        cell_w=cell_w,
        sprite_mask=sprite_mask,
        positions=positions,
        missing_idx=missing_idx,
    ):
        return None

    inp, out = _render_sequence(
        orientation=orientation,
        bg=bg,
        sep=sep,
        sprite_color=sprite_color,
        indicator_color=indicator_color,
        sprite_mask=sprite_mask,
        positions=positions,
        missing_idx=missing_idx,
        cell_h=cell_h,
        cell_w=cell_w,
    )

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        sample = _sample_case()
        if sample is not None:
            return sample[0]
    raise RuntimeError("failed to generate a valid input")
