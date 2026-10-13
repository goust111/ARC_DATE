import random

import numpy as np


BG = 0
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_K = {
    "top": 0,
    "right": 3,   # clockwise 90
    "bottom": 2,
    "left": 1,    # clockwise 270
}

MAX_GEN_ATTEMPTS = 300
GRID_MAX = 30


def _components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
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
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            pts = np.asarray(coords, dtype=int)
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            local[pts[:, 0] - r0, pts[:, 1] - c0] = True
            comps.append({
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "local": local,
            })
    return comps


def _classify_frame(local: np.ndarray, bbox: tuple[int, int, int, int]) -> dict | None:
    h, w = local.shape
    if h < 3 or w < 3:
        return None

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if np.any(local & ~border):
        return None

    missing = border & ~local
    missing_pts = np.argwhere(missing)
    if missing_pts.size == 0:
        if np.array_equal(local, border):
            return {"kind": "source", "bbox": bbox, "side": None}
        return None

    if len(missing_pts) != 1:
        return None

    rr, cc = map(int, missing_pts[0])
    on_h = rr == 0 or rr == h - 1
    on_v = cc == 0 or cc == w - 1
    if on_h == on_v:
        return None

    if rr == 0:
        side = "top"
    elif rr == h - 1:
        side = "bottom"
    elif cc == 0:
        side = "left"
    else:
        side = "right"
    return {"kind": "target", "bbox": bbox, "side": side}


def _rotate_for_side(arr: np.ndarray, side: str) -> np.ndarray:
    return np.rot90(arr, SIDE_TO_K[side])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair source and target rectangles by border color, rotate the source pattern
    according to the target gap side, move it into the target, and erase sources.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frame_infos: dict[int, list[dict]] = {}
    for color in sorted(int(x) for x in np.unique(g) if int(x) != BG):
        infos = []
        for comp in _components(g == color):
            info = _classify_frame(comp["local"], comp["bbox"])
            if info is not None:
                infos.append(info)
        if infos:
            frame_infos[color] = infos

    out = g.copy()
    for color, infos in frame_infos.items():
        assert len(infos) == 2, f"expected one source and one target for border color {color}"
        sources = [info for info in infos if info["kind"] == "source"]
        targets = [info for info in infos if info["kind"] == "target"]
        assert len(sources) == 1 and len(targets) == 1

        sr0, sc0, sr1, sc1 = sources[0]["bbox"]
        tr0, tc0, tr1, tc1 = targets[0]["bbox"]

        source_interior = g[sr0 + 1 : sr1, sc0 + 1 : sc1].copy()
        target_interior = g[tr0 + 1 : tr1, tc0 + 1 : tc1]
        assert source_interior.size > 0 and target_interior.size > 0
        assert np.any(source_interior != BG), "source must contain a pattern"
        assert np.all(target_interior == BG), "target interior must start empty"

        rotated = _rotate_for_side(source_interior, targets[0]["side"])
        assert rotated.shape == target_interior.shape, "rotation must fit the target interior"

        out[sr0 : sr1 + 1, sc0 : sc1 + 1] = BG
        out[tr0 + 1 : tr1, tc0 + 1 : tc1] = rotated

    return out


def _grow_connected_mask(h: int, w: int, area: int) -> np.ndarray | None:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _make_pattern(interior_size: int, color: int) -> np.ndarray:
    for _ in range(200):
        ph = random.randint(1, interior_size - 2)
        pw = random.randint(1, interior_size - 2)
        if ph * pw < 2:
            continue
        max_area = ph * pw if ph * pw <= 3 else ph * pw - 1
        area = random.randint(2, max_area)
        mask = _grow_connected_mask(ph, pw, area)
        if mask is None:
            continue
        r0 = random.randint(1, interior_size - ph - 1)
        c0 = random.randint(1, interior_size - pw - 1)
        arr = np.full((interior_size, interior_size), BG, dtype=int)
        arr[r0 : r0 + ph, c0 : c0 + pw][mask] = color
        if all(not np.array_equal(arr, np.rot90(arr, k)) for k in (1, 2, 3)):
            return arr
    raise RuntimeError("failed to build a non-degenerate pattern")


def _draw_frame_patch(border_color: int, interior: np.ndarray | None, side: str | None) -> np.ndarray:
    n = interior.shape[0] if interior is not None else side  # type: ignore[assignment]
    if isinstance(n, str):
        raise AssertionError("interior is required for source patches")
    patch = np.full((n + 2, n + 2), BG, dtype=int)
    patch[0, :] = border_color
    patch[-1, :] = border_color
    patch[:, 0] = border_color
    patch[:, -1] = border_color
    if interior is not None:
        patch[1:-1, 1:-1] = interior
    if side is not None:
        gap = random.randint(1, n)
        if side == "top":
            patch[0, gap] = BG
        elif side == "bottom":
            patch[-1, gap] = BG
        elif side == "left":
            patch[gap, 0] = BG
        else:
            patch[gap, -1] = BG
    return patch


def _layout_objects(objects: list[dict]) -> tuple[np.ndarray, dict[int, tuple[int, int]]] | None:
    shuffled = objects[:]
    random.shuffle(shuffled)
    rows = [shuffled[i : i + 2] for i in range(0, len(shuffled), 2)]
    row_specs: list[dict] = []
    total_h = 0
    total_w = 0
    v_gaps: list[int] = []

    for row in rows:
        gap = random.randint(1, 3) if len(row) == 2 else 0
        row_h = max(obj["patch"].shape[0] for obj in row)
        row_w = sum(obj["patch"].shape[1] for obj in row) + gap
        local_x = 0
        placements = []
        for idx, obj in enumerate(row):
            ph, pw = obj["patch"].shape
            y = random.randint(0, row_h - ph)
            placements.append((obj["id"], y, local_x))
            local_x += pw
            if idx == 0 and len(row) == 2:
                local_x += gap
        row_specs.append({"h": row_h, "w": row_w, "placements": placements, "objs": row})
        total_h += row_h
        total_w = max(total_w, row_w)

    for _ in range(len(rows) - 1):
        vg = random.randint(1, 3)
        v_gaps.append(vg)
        total_h += vg

    if total_h > GRID_MAX or total_w > GRID_MAX:
        return None

    grid_h = random.randint(total_h, min(GRID_MAX, total_h + 6))
    grid_w = random.randint(total_w, min(GRID_MAX, total_w + 8))
    top = random.randint(0, grid_h - total_h)
    left = random.randint(0, grid_w - total_w)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    positions: dict[int, tuple[int, int]] = {}
    cur_y = top
    for row_idx, spec in enumerate(row_specs):
        row_x = left + random.randint(0, total_w - spec["w"])
        for obj_id, off_y, off_x in spec["placements"]:
            patch = next(obj["patch"] for obj in spec["objs"] if obj["id"] == obj_id)
            ph, pw = patch.shape
            r0 = cur_y + off_y
            c0 = row_x + off_x
            grid[r0 : r0 + ph, c0 : c0 + pw] = np.where(patch != BG, patch, grid[r0 : r0 + ph, c0 : c0 + pw])
            positions[obj_id] = (r0, c0)
        cur_y += spec["h"]
        if row_idx < len(v_gaps):
            cur_y += v_gaps[row_idx]
    return grid, positions


def _expected_output(inp: np.ndarray, pair_specs: list[dict], positions: dict[int, tuple[int, int]]) -> np.ndarray:
    out = inp.copy()
    for spec in pair_specs:
        size = spec["interior_size"]
        sr, sc = positions[spec["source_id"]]
        tr, tc = positions[spec["target_id"]]
        out[sr : sr + size + 2, sc : sc + size + 2] = BG
        out[tr + 1 : tr + 1 + size, tc + 1 : tc + 1 + size] = _rotate_for_side(spec["pattern"], spec["side"])
    return out


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(2, 3)
    border_colors = random.sample(range(1, 10), pair_count)
    remaining = [c for c in range(1, 10) if c not in border_colors]
    pattern_colors = random.sample(remaining, pair_count)
    sides = random.sample(["top", "right", "bottom", "left"], pair_count)

    objects: list[dict] = []
    pair_specs: list[dict] = []

    next_id = 0
    for border_color, pattern_color, side in zip(border_colors, pattern_colors, sides):
        interior_size = random.randint(4, 6)
        pattern = _make_pattern(interior_size, pattern_color)
        source_patch = _draw_frame_patch(border_color, pattern, None)
        target_patch = _draw_frame_patch(border_color, np.full((interior_size, interior_size), BG, dtype=int), side)

        source_id = next_id
        target_id = next_id + 1
        next_id += 2

        objects.append({"id": source_id, "patch": source_patch})
        objects.append({"id": target_id, "patch": target_patch})
        pair_specs.append({
            "source_id": source_id,
            "target_id": target_id,
            "interior_size": interior_size,
            "pattern": pattern,
            "side": side,
        })

    placed = _layout_objects(objects)
    if placed is None:
        return None
    inp, positions = placed
    expected = _expected_output(inp, pair_specs, positions)

    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
