from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
BRIDGE = 8
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SPRITE_PALETTE = [1, 2, 3, 4, 5, 6, 7, 9]
SIDES = ("top", "bottom", "left", "right")


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(arr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(
    dst: np.ndarray,
    patch: np.ndarray,
    r0: int,
    c0: int,
    *,
    transparent: int | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    h, w = patch.shape
    rr0, cc0 = int(r0), int(c0)
    rr1, cc1 = rr0 + h, cc0 + w
    if rr0 < 0 or cc0 < 0 or rr1 > dst.shape[0] or cc1 > dst.shape[1]:
        raise ValueError("paste out of bounds")
    dst_view = dst[rr0:rr1, cc0:cc1]
    use = np.ones((h, w), dtype=bool)
    if transparent is not None:
        use &= patch != int(transparent)
    if mask is not None:
        use &= mask.astype(bool)
    dst_view[use] = patch[use]
    return dst


def _label_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    lab = np.zeros(mask.shape, dtype=int)
    n = 0
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or lab[r, c] != 0:
                continue
            n += 1
            q = deque([(r, c)])
            lab[r, c] = n
            while q:
                rr, cc = q.popleft()
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and lab[nr, nc] == 0:
                        lab[nr, nc] = n
                        q.append((nr, nc))
    return lab, n


def _component_dicts(mask: np.ndarray) -> list[dict]:
    lab, n = _label_mask(mask)
    out = []
    for idx in range(1, n + 1):
        coords = np.argwhere(lab == idx)
        out.append({"coords": coords, "bbox": _bbox_coords(coords), "area": int(coords.shape[0])})
    return out


def _find_hub(grid: np.ndarray, bg: int) -> dict:
    h, w = grid.shape
    best = None
    for n in range(4, min(9, h, w) + 1):
        for r0 in range(h - n + 1):
            for c0 in range(w - n + 1):
                if (
                    int(grid[r0, c0]) != bg
                    or int(grid[r0, c0 + n - 1]) != bg
                    or int(grid[r0 + n - 1, c0]) != bg
                    or int(grid[r0 + n - 1, c0 + n - 1]) != bg
                ):
                    continue
                if not np.all(grid[r0 + 1 : r0 + n - 1, c0 + 1 : c0 + n - 1] == bg):
                    continue
                top = grid[r0, c0 + 1 : c0 + n - 1]
                bottom = grid[r0 + n - 1, c0 + 1 : c0 + n - 1]
                left = grid[r0 + 1 : r0 + n - 1, c0]
                right = grid[r0 + 1 : r0 + n - 1, c0 + n - 1]
                edges = [top, bottom, left, right]
                if any(edge.size == 0 or np.any(edge == bg) for edge in edges):
                    continue
                cols = [int(top[0]), int(bottom[0]), int(left[0]), int(right[0])]
                if BRIDGE in cols or len(set(cols)) != 4:
                    continue
                if any(len(set(map(int, edge))) != 1 for edge in edges):
                    continue
                center_score = abs((r0 + (n - 1) / 2) - (h - 1) / 2) + abs((c0 + (n - 1) / 2) - (w - 1) / 2)
                score = (center_score, -n)
                if best is None or score < best["score"]:
                    edge_mask = np.zeros_like(grid, dtype=bool)
                    edge_mask[r0, c0 + 1 : c0 + n - 1] = True
                    edge_mask[r0 + n - 1, c0 + 1 : c0 + n - 1] = True
                    edge_mask[r0 + 1 : r0 + n - 1, c0] = True
                    edge_mask[r0 + 1 : r0 + n - 1, c0 + n - 1] = True
                    square_mask = np.zeros_like(grid, dtype=bool)
                    square_mask[r0 : r0 + n, c0 : c0 + n] = True
                    best = {
                        "score": score,
                        "r": int(r0),
                        "c": int(c0),
                        "n": int(n),
                        "edge_mask": edge_mask,
                        "square_mask": square_mask,
                        "colors": {
                            "top": int(top[0]),
                            "bottom": int(bottom[0]),
                            "left": int(left[0]),
                            "right": int(right[0]),
                        },
                    }
    if best is None:
        raise AssertionError("hub not found")
    return best


def _edge_segment(mask: np.ndarray, side: str) -> tuple[int, list[int]] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    if side == "top":
        row = int(coords[:, 0].min())
        values = sorted(int(c) for c in coords[coords[:, 0] == row][:, 1])
        return row, values
    if side == "bottom":
        row = int(coords[:, 0].max())
        values = sorted(int(c) for c in coords[coords[:, 0] == row][:, 1])
        return row, values
    if side == "left":
        col = int(coords[:, 1].min())
        values = sorted(int(r) for r in coords[coords[:, 1] == col][:, 0])
        return col, values
    col = int(coords[:, 1].max())
    values = sorted(int(r) for r in coords[coords[:, 1] == col][:, 0])
    return col, values


def _is_contiguous(vals: list[int]) -> bool:
    return bool(vals) and all(vals[i] + 1 == vals[i + 1] for i in range(len(vals) - 1))


def _docking_candidates(
    sprite_patch: np.ndarray,
    side: str,
    target_len: int,
    hub: dict,
    out: np.ndarray,
    bg: int,
) -> list[tuple[np.ndarray, np.ndarray, int, int]]:
    r0, c0, n = hub["r"], hub["c"], hub["n"]
    spans = {
        "top": {"need": "bottom", "fixed": r0 - 1, "span": list(range(c0 + 1, c0 + n - 1))},
        "bottom": {"need": "top", "fixed": r0 + n, "span": list(range(c0 + 1, c0 + n - 1))},
        "left": {"need": "right", "fixed": c0 - 1, "span": list(range(r0 + 1, r0 + n - 1))},
        "right": {"need": "left", "fixed": c0 + n, "span": list(range(r0 + 1, r0 + n - 1))},
    }
    spec = spans[side]
    valid = []
    for k in range(4):
        rot = np.rot90(sprite_patch, k)
        rot_mask = rot != bg
        bbox = _bbox_coords(np.argwhere(rot_mask))
        rot = _crop_bbox(rot, bbox)
        rot_mask = rot != bg

        if side in ("left", "right"):
            if not np.array_equal(rot_mask, np.flipud(rot_mask)):
                continue
        else:
            if not np.array_equal(rot_mask, np.fliplr(rot_mask)):
                continue

        edge = _edge_segment(rot_mask, spec["need"])
        if edge is None:
            continue
        fixed_coord, values = edge
        if len(values) != target_len or not _is_contiguous(values):
            continue

        if side in ("left", "right"):
            dr = spec["span"][0] - values[0]
            dc = spec["fixed"] - fixed_coord
        else:
            dr = spec["fixed"] - fixed_coord
            dc = spec["span"][0] - values[0]

        coords = np.argwhere(rot_mask)
        placed = coords + np.array([[dr, dc]])
        if np.any(placed[:, 0] < 0) or np.any(placed[:, 0] >= out.shape[0]) or np.any(placed[:, 1] < 0) or np.any(placed[:, 1] >= out.shape[1]):
            continue
        ok = True
        for rr, cc in placed:
            if int(out[rr, cc]) not in (bg, int(rot[rr - dr, cc - dc])):
                ok = False
                break
        if ok:
            valid.append((rot, rot_mask, int(dr), int(dc)))
    return valid


def _sprite_by_side(grid: np.ndarray, hub: dict) -> dict[str, dict]:
    side_map: dict[str, dict] = {}
    for side, color in hub["colors"].items():
        mask = (grid == color) & (~hub["square_mask"])
        comps = _component_dicts(mask)
        if not comps:
            continue
        chosen = max(comps, key=lambda comp: comp["area"])
        sprite_mask = np.zeros_like(grid, dtype=bool)
        sprite_mask[tuple(chosen["coords"].T)] = True
        patch = np.full_like(grid, BG)
        patch[sprite_mask] = color
        patch = _crop_bbox(patch, chosen["bbox"])
        side_map[side] = {
            "color": color,
            "mask": sprite_mask,
            "patch": patch,
        }
    return side_map


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the central hollow square and dock only the sprites that are bridge-connected to it.

    Input structure:
    - The grid has a black background, one central hollow square whose four edges are four distinct colors, and 2-4 scattered monochrome sprites that reuse those edge colors.
    - Teal bridge pixels form 1-cell-wide 4-connected chains; some chains connect sprites to the hub, while at least one sprite has no bridge connection.
    - Hub corners and interior are black, and sprite colors are distinct from the teal bridge color.

    Rule:
    - Detect the hollow square hub and treat all non-black pixels as one 4-connected graph, ignoring color differences.
    - For each sprite whose component touches any hub edge component, match its color to the corresponding hub side, rotate it so the contact edge has exactly N-2 contiguous cells and the rotated sprite has the required mirror symmetry, then dock it flush against that side.
    - Erase every sprite that is not connected to the hub and erase all teal bridge pixels; keep only the hub and the successfully docked sprites on the original canvas.

    Key insight:
    - The teal paths are invisible credentials: they do not survive in the output, but they decide which color-matched sprites are allowed to participate in the same docking rule as the hollow-square puzzle.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    hub = _find_hub(g, bg)

    out = np.full_like(g, bg)
    out[hub["edge_mask"]] = g[hub["edge_mask"]]

    fg_labels, _ = _label_mask(g != bg)
    hub_labels = {int(v) for v in fg_labels[hub["edge_mask"]] if int(v) > 0}
    target_len = int(hub["n"] - 2)

    for side, sprite in _sprite_by_side(g, hub).items():
        sprite_labels = {int(v) for v in fg_labels[sprite["mask"]] if int(v) > 0}
        if not sprite_labels & hub_labels:
            continue
        choices = _docking_candidates(sprite["patch"], side, target_len, hub, out, bg)
        if not choices:
            raise AssertionError(f"no docking rotation for {side}")
        rot, rot_mask, dr, dc = choices[0]
        _paste(out, rot, dr, dc, mask=rot_mask)

    return out


def _canonical_left_masks(edge_len: int) -> list[np.ndarray]:
    if edge_len == 3:
        return [
            np.array([[0, 1], [1, 1], [0, 1]], dtype=int),
            np.array([[1, 1], [0, 1], [1, 1]], dtype=int),
            np.array([[0, 0, 1], [1, 1, 1], [0, 0, 1]], dtype=int),
        ]
    if edge_len == 4:
        return [
            np.array([[0, 1], [1, 1], [1, 1], [0, 1]], dtype=int),
            np.array([[0, 0, 1], [1, 1, 1], [1, 1, 1], [0, 0, 1]], dtype=int),
        ]
    if edge_len == 5:
        return [
            np.array([[0, 1], [0, 1], [1, 1], [0, 1], [0, 1]], dtype=int),
            np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1]], dtype=int),
        ]
    raise ValueError("unsupported edge length")


def _oriented_mask(edge_len: int, side: str, rng: random.Random) -> np.ndarray:
    options = _canonical_left_masks(edge_len)
    base = options[rng.randrange(len(options))]
    if side == "left":
        mask = base
    elif side == "right":
        mask = np.fliplr(base)
    elif side == "top":
        mask = np.rot90(base, -1)
    else:
        mask = np.rot90(base, 1)
    return mask.astype(bool)


def _build_hub(shape: tuple[int, int], r0: int, c0: int, n: int, side_colors: dict[str, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    grid[r0, c0 + 1 : c0 + n - 1] = side_colors["top"]
    grid[r0 + n - 1, c0 + 1 : c0 + n - 1] = side_colors["bottom"]
    grid[r0 + 1 : r0 + n - 1, c0] = side_colors["left"]
    grid[r0 + 1 : r0 + n - 1, c0 + n - 1] = side_colors["right"]
    return grid


def _occupied_coords(mask: np.ndarray, r0: int, c0: int) -> set[tuple[int, int]]:
    coords = np.argwhere(mask)
    return {(int(r0 + r), int(c0 + c)) for r, c in coords}


def _docked_location(mask: np.ndarray, side: str, hub_r: int, hub_c: int, hub_n: int) -> tuple[int, int]:
    if side == "top":
        cols = np.where(mask[-1])[0]
        return hub_r - mask.shape[0], hub_c + 1 - int(cols[0])
    if side == "bottom":
        cols = np.where(mask[0])[0]
        return hub_r + hub_n, hub_c + 1 - int(cols[0])
    if side == "left":
        rows = np.where(mask[:, -1])[0]
        return hub_r + 1 - int(rows[0]), hub_c - mask.shape[1]
    rows = np.where(mask[:, 0])[0]
    return hub_r + 1 - int(rows[0]), hub_c + hub_n


def _can_place(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, padding: int) -> bool:
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > grid.shape[0] or c0 + w > grid.shape[1]:
        return False
    occ = np.argwhere(patch != BG)
    for pr, pc in occ:
        rr, cc = int(r0 + pr), int(c0 + pc)
        if int(grid[rr, cc]) != BG:
            return False
        for dr in range(-padding, padding + 1):
            for dc in range(-padding, padding + 1):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) != BG:
                    return False
    return True


def _random_free_location(grid: np.ndarray, patch: np.ndarray, rng: random.Random, *, border: int = 1, padding: int = 1) -> tuple[int, int] | None:
    h, w = patch.shape
    candidates = []
    for r0 in range(border, grid.shape[0] - h - border + 1):
        for c0 in range(border, grid.shape[1] - w - border + 1):
            if _can_place(grid, patch, r0, c0, padding):
                candidates.append((int(r0), int(c0)))
    if not candidates:
        return None
    return candidates[rng.randrange(len(candidates))]


def _neighbor_nonblack(grid: np.ndarray, cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) != BG:
            out.append((nr, nc))
    return out


def _bridge_path(
    grid: np.ndarray,
    sprite_coords: set[tuple[int, int]],
    hub_coords: set[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    starts = []
    goals = set()
    for r, c in sprite_coords:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) == BG:
                touched = set(_neighbor_nonblack(grid, (nr, nc)))
                if touched and touched <= sprite_coords:
                    starts.append((nr, nc))

    for r, c in hub_coords:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) == BG:
                touched = set(_neighbor_nonblack(grid, (nr, nc)))
                if touched and touched <= hub_coords:
                    goals.add((nr, nc))

    if not starts or not goals:
        return None

    rng.shuffle(starts)
    goal_set = set(goals)

    def is_clear(cell: tuple[int, int]) -> bool:
        if cell in goal_set:
            return True
        if int(grid[cell]) != BG:
            return False
        return len(_neighbor_nonblack(grid, cell)) == 0

    q = deque()
    parent: dict[tuple[int, int], tuple[int, int] | None] = {}
    for cell in starts:
        if cell in parent:
            continue
        parent[cell] = None
        q.append(cell)

    while q:
        cur = q.popleft()
        if cur in goal_set:
            path = []
            node = cur
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return path
        dirs = DIR4[:]
        rng.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = cur[0] + dr, cur[1] + dc
            nxt = (nr, nc)
            if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]):
                continue
            if nxt in parent:
                continue
            if is_clear(nxt):
                parent[nxt] = cur
                q.append(nxt)
    return None


def _center_distance(coords: set[tuple[int, int]], center: tuple[float, float]) -> float:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    cr = (min(rows) + max(rows)) / 2
    cc = (min(cols) + max(cols)) / 2
    return abs(cr - center[0]) + abs(cc - center[1])


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(18, 24)
    w = rng.randint(18, 24)
    hub_n = rng.randint(5, 7)
    edge_len = hub_n - 2
    side_colors_list = rng.sample(SPRITE_PALETTE, 4)
    side_colors = dict(zip(SIDES, side_colors_list))

    max_margin = 5
    if h - hub_n - 2 * max_margin < 0 or w - hub_n - 2 * max_margin < 0:
        return None
    hub_r = rng.randint(max_margin, h - hub_n - max_margin)
    hub_c = rng.randint(max_margin, w - hub_n - max_margin)

    chosen_sides = rng.sample(list(SIDES), rng.randint(2, 4))
    connected_count = rng.randint(1, len(chosen_sides) - 1)
    connected_sides = set(rng.sample(chosen_sides, connected_count))

    output_grid = _build_hub((h, w), hub_r, hub_c, hub_n, side_colors)
    input_grid = output_grid.copy()
    hub_mask = output_grid != BG
    hub_coords = {tuple(coord) for coord in np.argwhere(hub_mask)}

    sprite_specs = []
    for side in chosen_sides:
        dock_mask = _oriented_mask(edge_len, side, rng)
        color = side_colors[side]
        dock_patch = np.where(dock_mask, color, BG)
        dock_r, dock_c = _docked_location(dock_mask, side, hub_r, hub_c, hub_n)
        if side in connected_sides:
            if not _can_place(output_grid, dock_patch, dock_r, dock_c, padding=0):
                return None
            _paste(output_grid, dock_patch, dock_r, dock_c, transparent=BG)

        input_patch = dock_patch.copy()
        k = rng.randrange(4)
        if side in connected_sides and k == 0:
            k = rng.randint(1, 3)
        input_patch = np.rot90(input_patch, k)
        bbox = _bbox_coords(np.argwhere(input_patch != BG))
        input_patch = _crop_bbox(input_patch, bbox)
        loc = _random_free_location(input_grid, input_patch, rng, border=1, padding=1)
        if loc is None:
            return None
        _paste(input_grid, input_patch, loc[0], loc[1], transparent=BG)
        sprite_specs.append(
            {
                "side": side,
                "connected": side in connected_sides,
                "coords": _occupied_coords(input_patch != BG, loc[0], loc[1]),
            }
        )

    connected_dists = []
    disconnected_dists = []
    hub_center = (hub_r + (hub_n - 1) / 2, hub_c + (hub_n - 1) / 2)
    for spec in sprite_specs:
        dist = _center_distance(spec["coords"], hub_center)
        if spec["connected"]:
            connected_dists.append(dist)
        else:
            disconnected_dists.append(dist)
    if not connected_dists or not disconnected_dists or min(disconnected_dists) >= max(connected_dists):
        return None

    for spec in sprite_specs:
        if not spec["connected"]:
            continue
        path = _bridge_path(input_grid, spec["coords"], hub_coords, rng)
        if path is None:
            return None
        for r, c in path:
            input_grid[r, c] = BRIDGE

    try:
        if np.array_equal(main(input_grid), output_grid):
            return input_grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bridge-gated docking puzzle")
