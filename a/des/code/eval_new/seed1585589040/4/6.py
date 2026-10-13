from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np


SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
DELTAS = {"top": (-1, 0), "right": (0, 1), "bottom": (1, 0), "left": (0, -1)}
LAYOUTS = ((2, 2), (2, 3), (3, 2), (1, 5), (5, 1))
GEN_RETRIES = 400
PLACE_RETRIES = 300


def _components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] == 0:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [pt[0] for pt in coords]
            cols = [pt[1] for pt in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            comps.append({"color": color, "mask": mask})
    return comps


def _valid_offsets(total: int, core: int) -> list[int]:
    extra = total - core
    if extra < 0 or extra > 2:
        return []
    out = set()
    for before in (0, 1):
        after = extra - before
        if 0 <= after <= 1:
            out.add(before)
    return sorted(out)


def _explain_mask(mask: np.ndarray, core_h: int, core_w: int, color: int) -> list[dict]:
    bh, bw = mask.shape
    explanations: list[dict] = []
    for top_off in _valid_offsets(bh, core_h):
        for left_off in _valid_offsets(bw, core_w):
            if not _matches_core_model(mask, core_h, core_w, top_off, left_off):
                continue
            sigs = _extract_signatures(mask, core_h, core_w, top_off, left_off)
            if sigs is None:
                continue
            explanations.append(
                {
                    "color": color,
                    "mask": mask,
                    "top_offset": top_off,
                    "left_offset": left_off,
                    **sigs,
                }
            )
    return explanations


def _matches_core_model(mask: np.ndarray, core_h: int, core_w: int, top_off: int, left_off: int) -> bool:
    bh, bw = mask.shape
    bottom = top_off + core_h
    right = left_off + core_w
    for r in range(bh):
        for c in range(bw):
            inside = top_off <= r < bottom and left_off <= c < right
            if inside:
                if top_off < r < bottom - 1 and left_off < c < right - 1 and not mask[r, c]:
                    return False
            elif mask[r, c]:
                allowed = (
                    (r == top_off - 1 and left_off <= c < right)
                    or (r == bottom and left_off <= c < right)
                    or (c == left_off - 1 and top_off <= r < bottom)
                    or (c == right and top_off <= r < bottom)
                )
                if not allowed:
                    return False
    return True


def _extract_signatures(mask: np.ndarray, core_h: int, core_w: int, top_off: int, left_off: int) -> dict | None:
    bh, bw = mask.shape
    bottom = top_off + core_h
    right = left_off + core_w
    top = np.zeros(core_w, dtype=int)
    bottom_sig = np.zeros(core_w, dtype=int)
    left = np.zeros(core_h, dtype=int)
    right_sig = np.zeros(core_h, dtype=int)

    for j in range(core_w):
        boundary = bool(mask[top_off, left_off + j])
        protrusion = top_off > 0 and bool(mask[top_off - 1, left_off + j])
        if protrusion and not boundary:
            return None
        top[j] = 1 if protrusion else (-1 if not boundary else 0)

        boundary = bool(mask[bottom - 1, left_off + j])
        protrusion = bottom < bh and bool(mask[bottom, left_off + j])
        if protrusion and not boundary:
            return None
        bottom_sig[j] = 1 if protrusion else (-1 if not boundary else 0)

    for i in range(core_h):
        boundary = bool(mask[top_off + i, left_off])
        protrusion = left_off > 0 and bool(mask[top_off + i, left_off - 1])
        if protrusion and not boundary:
            return None
        left[i] = 1 if protrusion else (-1 if not boundary else 0)

        boundary = bool(mask[top_off + i, right - 1])
        protrusion = right < bw and bool(mask[top_off + i, right])
        if protrusion and not boundary:
            return None
        right_sig[i] = 1 if protrusion else (-1 if not boundary else 0)

    return {"top": top, "right": right_sig, "bottom": bottom_sig, "left": left}


def _complement(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.any(a != 0) and np.array_equal(a, -b)


def _add_neighbor(neighbors: list[dict], src: int, side: str, dst: int) -> bool:
    opp = OPPOSITE[side]
    if side in neighbors[src] and neighbors[src][side] != dst:
        return False
    if opp in neighbors[dst] and neighbors[dst][opp] != src:
        return False
    neighbors[src][side] = dst
    neighbors[dst][opp] = src
    return True


def _layout_from_explanations(exps: tuple[dict, ...]) -> tuple[list[dict], dict[int, tuple[int, int]], int, int] | None:
    n = len(exps)
    neighbors: list[dict] = [dict() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _complement(exps[i]["right"], exps[j]["left"]) and not _add_neighbor(neighbors, i, "right", j):
                return None
            if _complement(exps[j]["right"], exps[i]["left"]) and not _add_neighbor(neighbors, j, "right", i):
                return None
            if _complement(exps[i]["bottom"], exps[j]["top"]) and not _add_neighbor(neighbors, i, "bottom", j):
                return None
            if _complement(exps[j]["bottom"], exps[i]["top"]) and not _add_neighbor(neighbors, j, "bottom", i):
                return None

    for i, exp in enumerate(exps):
        for side in SIDES:
            if bool(np.any(exp[side] != 0)) != (side in neighbors[i]):
                return None

    pos = {0: (0, 0)}
    q = deque([0])
    while q:
        cur = q.popleft()
        cr, cc = pos[cur]
        for side, nxt in neighbors[cur].items():
            dr, dc = DELTAS[side]
            target = (cr + dr, cc + dc)
            if nxt in pos:
                if pos[nxt] != target:
                    return None
            else:
                pos[nxt] = target
                q.append(nxt)

    if len(pos) != n:
        return None

    min_r = min(r for r, _ in pos.values())
    min_c = min(c for _, c in pos.values())
    pos = {idx: (r - min_r, c - min_c) for idx, (r, c) in pos.items()}
    rows = max(r for r, _ in pos.values()) + 1
    cols = max(c for _, c in pos.values()) + 1
    if rows * cols != n:
        return None
    if set(pos.values()) != {(r, c) for r in range(rows) for c in range(cols)}:
        return None
    return neighbors, pos, rows, cols


def _reconstruct(grid: np.ndarray) -> tuple[tuple[dict, ...], list[dict], dict[int, tuple[int, int]], int, int, int, int]:
    tiles = _components(grid)
    assert 4 <= len(tiles) <= 6, "expected 4 to 6 tiles"

    h_low = max(max(1, tile["mask"].shape[0] - 2) for tile in tiles)
    h_high = min(tile["mask"].shape[0] for tile in tiles)
    w_low = max(max(1, tile["mask"].shape[1] - 2) for tile in tiles)
    w_high = min(tile["mask"].shape[1] for tile in tiles)

    for core_h in range(h_low, h_high + 1):
        for core_w in range(w_low, w_high + 1):
            candidate_lists = [_explain_mask(tile["mask"], core_h, core_w, tile["color"]) for tile in tiles]
            if any(not lst for lst in candidate_lists):
                continue
            for combo in itertools.product(*candidate_lists):
                solved = _layout_from_explanations(combo)
                if solved is not None:
                    neighbors, pos, rows, cols = solved
                    return combo, neighbors, pos, rows, cols, core_h, core_w
    raise AssertionError("failed to reconstruct a unique rectangular assembly")


def _render_output(
    exps: tuple[dict, ...],
    neighbors: list[dict],
    pos: dict[int, tuple[int, int]],
    rows: int,
    cols: int,
    core_h: int,
    core_w: int,
) -> np.ndarray:
    out = np.zeros((rows * core_h, cols * core_w), dtype=int)
    for idx, exp in enumerate(exps):
        base_r = pos[idx][0] * core_h
        base_c = pos[idx][1] * core_w
        top_off = int(exp["top_offset"])
        left_off = int(exp["left_offset"])
        mask = exp["mask"]
        color = int(exp["color"])
        for lr, lc in np.argwhere(mask):
            gr = base_r + int(lr) - top_off
            gc = base_c + int(lc) - left_off
            if not (0 <= gr < out.shape[0] and 0 <= gc < out.shape[1]):
                raise AssertionError("piece protruded outside the target rectangle")
            if out[gr, gc] not in (0, color):
                raise AssertionError("pieces overlap in assembled output")
            out[gr, gc] = color

    for idx, exp in enumerate(exps):
        base_r = pos[idx][0] * core_h
        base_c = pos[idx][1] * core_w
        if "right" in neighbors[idx]:
            right_idx = neighbors[idx]["right"]
            seam_color = int(exps[right_idx]["color"])
            for off, val in enumerate(exp["right"]):
                if val == 0:
                    continue
                rr = base_r + off
                cc = base_c + core_w if val == 1 else base_c + core_w - 1
                out[rr, cc] = seam_color
        if "bottom" in neighbors[idx]:
            bottom_idx = neighbors[idx]["bottom"]
            seam_color = int(exps[bottom_idx]["color"])
            for off, val in enumerate(exp["bottom"]):
                if val == 0:
                    continue
                rr = base_r + core_h if val == 1 else base_r + core_h - 1
                cc = base_c + off
                out[rr, cc] = seam_color

    assert np.all(out != 0), "assembled rectangle must be fully filled"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered monochrome jigsaw tiles into a full rectangle and recolor each interlock seam with the later adjacent tile.

    Input structure:
    - The background is black.
    - The foreground contains 4 to 6 disconnected monochrome tiles, each with a distinct color.
    - Every tile shares the same hidden core rectangle size, but its border may contain one-cell protrusions or one-cell notches along some sides.

    Rule:
    - Recover each tile's four edge signatures by comparing its shape to the shared core rectangle.
    - Match edges only when one side's protrusion pattern is the exact complement of the other side's notch pattern, then translate all tiles into the unique rectangle whose cells are fully filled by these matches.
    - After assembly, recolor every tooth pixel used on a seam to the color of the adjacent tile that appears later in the final row-major layout: the right tile across vertical seams and the lower tile across horizontal seams.

    Key insight:
    - The puzzle is solved by first inferring hidden adjacency from complementary border geometry, then using the assembled spatial order to decide the seam color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    exps, neighbors, pos, rows, cols, core_h, core_w = _reconstruct(g)
    return _render_output(exps, neighbors, pos, rows, cols, core_h, core_w)


def _nonempty_supports(length: int) -> list[tuple[int, ...]]:
    interior = list(range(1, length - 1))
    supports: list[tuple[int, ...]] = []
    for mask in range(1, 1 << len(interior)):
        subset = [interior[i] for i in range(len(interior)) if mask & (1 << i)]
        supports.append(tuple(subset))
    return supports


def _min_core_length(seam_count: int) -> int:
    length = 4
    while (1 << max(0, length - 2)) - 1 < seam_count:
        length += 1
    return length


def _mask_from_signatures(top: np.ndarray, right: np.ndarray, bottom: np.ndarray, left: np.ndarray) -> tuple[np.ndarray, int, int]:
    core_h = len(left)
    core_w = len(top)
    top_ext = int(np.any(top == 1))
    right_ext = int(np.any(right == 1))
    bottom_ext = int(np.any(bottom == 1))
    left_ext = int(np.any(left == 1))
    mask = np.zeros((core_h + top_ext + bottom_ext, core_w + left_ext + right_ext), dtype=bool)
    top_off = top_ext
    left_off = left_ext
    mask[top_off : top_off + core_h, left_off : left_off + core_w] = True

    mask[top_off, left_off + np.where(top == -1)[0]] = False
    mask[top_off + core_h - 1, left_off + np.where(bottom == -1)[0]] = False
    mask[top_off + np.where(left == -1)[0], left_off] = False
    mask[top_off + np.where(right == -1)[0], left_off + core_w - 1] = False

    if top_ext:
        mask[top_off - 1, left_off + np.where(top == 1)[0]] = True
    if bottom_ext:
        mask[top_off + core_h, left_off + np.where(bottom == 1)[0]] = True
    if left_ext:
        mask[top_off + np.where(left == 1)[0], left_off - 1] = True
    if right_ext:
        mask[top_off + np.where(right == 1)[0], left_off + core_w] = True
    return mask, top_off, left_off


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    seen = np.zeros_like(mask, dtype=bool)
    start = tuple(map(int, pts[0]))
    q = deque([start])
    seen[start] = True
    count = 0
    while q:
        r, c = q.popleft()
        count += 1
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return count == int(mask.sum())


def _build_blueprint(rng: random.Random) -> tuple[tuple[dict, ...], list[dict], dict[int, tuple[int, int]], int, int, int, int]:
    rows, cols = rng.choice(LAYOUTS)
    vertical_seams = rows * max(0, cols - 1)
    horizontal_seams = max(0, rows - 1) * cols
    core_h = rng.randint(_min_core_length(vertical_seams), min(6, 30 // rows))
    core_w = rng.randint(_min_core_length(horizontal_seams), min(6, 30 // cols))

    edge_grid: list[list[dict[str, np.ndarray]]] = []
    for _ in range(rows):
        row: list[dict[str, np.ndarray]] = []
        for _ in range(cols):
            row.append(
                {
                    "top": np.zeros(core_w, dtype=int),
                    "right": np.zeros(core_h, dtype=int),
                    "bottom": np.zeros(core_w, dtype=int),
                    "left": np.zeros(core_h, dtype=int),
                }
            )
        edge_grid.append(row)

    vertical_patterns = _nonempty_supports(core_h)
    horizontal_patterns = _nonempty_supports(core_w)
    rng.shuffle(vertical_patterns)
    rng.shuffle(horizontal_patterns)

    for r in range(rows):
        for c in range(cols - 1):
            subset = vertical_patterns.pop()
            pattern = np.zeros(core_h, dtype=int)
            pattern[list(subset)] = 1
            if rng.random() < 0.5:
                edge_grid[r][c]["right"] = pattern
                edge_grid[r][c + 1]["left"] = -pattern
            else:
                edge_grid[r][c]["right"] = -pattern
                edge_grid[r][c + 1]["left"] = pattern

    for r in range(rows - 1):
        for c in range(cols):
            subset = horizontal_patterns.pop()
            pattern = np.zeros(core_w, dtype=int)
            pattern[list(subset)] = 1
            if rng.random() < 0.5:
                edge_grid[r][c]["bottom"] = pattern
                edge_grid[r + 1][c]["top"] = -pattern
            else:
                edge_grid[r][c]["bottom"] = -pattern
                edge_grid[r + 1][c]["top"] = pattern

    colors = rng.sample(list(range(1, 10)), rows * cols)
    exps: list[dict] = []
    pos: dict[int, tuple[int, int]] = {}
    idx = 0
    for r in range(rows):
        for c in range(cols):
            sigs = edge_grid[r][c]
            mask, top_off, left_off = _mask_from_signatures(sigs["top"], sigs["right"], sigs["bottom"], sigs["left"])
            if not _is_connected(mask):
                raise AssertionError("generated a disconnected tile")
            exps.append(
                {
                    "color": colors[idx],
                    "mask": mask,
                    "top_offset": top_off,
                    "left_offset": left_off,
                    "top": sigs["top"],
                    "right": sigs["right"],
                    "bottom": sigs["bottom"],
                    "left": sigs["left"],
                }
            )
            pos[idx] = (r, c)
            idx += 1

    combo = tuple(exps)
    neighbors, _, _, _ = _layout_from_explanations(combo)  # type: ignore[misc]
    return combo, neighbors, pos, rows, cols, core_h, core_w


def _place_scattered(exps: tuple[dict, ...], rng: random.Random) -> np.ndarray | None:
    max_h = max(exp["mask"].shape[0] for exp in exps)
    max_w = max(exp["mask"].shape[1] for exp in exps)
    total_area = sum(int(exp["mask"].sum()) for exp in exps)
    min_side = max(max_h + 2, max_w + 2, int(np.ceil(np.sqrt(total_area * 3.0))))
    grid_h = rng.randint(max(18, min_side), 30)
    grid_w = rng.randint(max(18, min_side), 30)

    order = list(range(len(exps)))
    rng.shuffle(order)
    order.sort(key=lambda idx: int(exps[idx]["mask"].sum()), reverse=True)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    placements: dict[int, tuple[int, int]] = {}

    for idx in order:
        mask = exps[idx]["mask"]
        bh, bw = mask.shape
        placed = False
        for _ in range(PLACE_RETRIES):
            r = rng.randint(0, grid_h - bh)
            c = rng.randint(0, grid_w - bw)
            r0 = max(0, r - 1)
            c0 = max(0, c - 1)
            r1 = min(grid_h, r + bh + 1)
            c1 = min(grid_w, c + bw + 1)
            if blocked[r0:r1, c0:c1].any():
                continue
            placements[idx] = (r, c)
            blocked[r0:r1, c0:c1] = True
            placed = True
            break
        if not placed:
            return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for idx, (r, c) in placements.items():
        mask = exps[idx]["mask"]
        color = int(exps[idx]["color"])
        patch = grid[r : r + mask.shape[0], c : c + mask.shape[1]]
        patch[mask] = color
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            exps, neighbors, pos, rows, cols, core_h, core_w = _build_blueprint(rng)
        except Exception:
            continue
        grid = _place_scattered(exps, rng)
        if grid is None:
            continue
        try:
            expected = _render_output(exps, neighbors, pos, rows, cols, core_h, core_w)
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
