import random
from typing import Any

import numpy as np


BG = 0
AZURE = 8

ROT4 = "rot4"
MIRROR_LR = "mirror_lr"
MIRROR_UD = "mirror_ud"

CORNER_ORDER = ("TL", "TR", "BR", "BL")
HORIZONTAL_OPPOSITE = {"TL": "TR", "TR": "TL", "BL": "BR", "BR": "BL"}
VERTICAL_OPPOSITE = {"TL": "BL", "BL": "TL", "TR": "BR", "BR": "TR"}

FRAME_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
MARKER_TYPES = (ROT4, MIRROR_LR, MIRROR_UD)
GEN_RETRIES = 400
GRID_H_RANGE = (20, 33)
GRID_W_RANGE = (20, 33)
FRAME_COUNT_CHOICES = (1, 2, 3)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def _find_frames(grid: np.ndarray) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (BG, AZURE)):
        for coords in _components(grid == color):
            r0, r1, c0, c1 = _bbox(coords)
            hh = r1 - r0 + 1
            ww = c1 - c0 + 1
            if hh < 3 or ww < 3:
                continue
            sub = grid[r0 : r1 + 1, c0 : c1 + 1] == color
            ring = np.zeros((hh, ww), dtype=bool)
            ring[0, :] = True
            ring[-1, :] = True
            ring[:, 0] = True
            ring[:, -1] = True
            if not np.array_equal(sub, ring):
                continue
            frames.append(
                {
                    "color": color,
                    "r0": r0,
                    "r1": r1,
                    "c0": c0,
                    "c1": c1,
                    "inner_h": hh - 2,
                    "inner_w": ww - 2,
                }
            )
    frames.sort(key=lambda frame: (frame["r0"], frame["c0"]))
    return frames


def _marker_kind(grid: np.ndarray, frame: dict[str, Any]) -> str:
    interior = grid[frame["r0"] + 1 : frame["r1"], frame["c0"] + 1 : frame["c1"]]
    marker = set(map(tuple, np.argwhere(interior == AZURE).tolist()))
    cr = (interior.shape[0] - 1) // 2
    cc = (interior.shape[1] - 1) // 2
    if marker == {(cr, cc)}:
        return ROT4
    if cc + 1 < interior.shape[1] and marker == {(cr, cc), (cr, cc + 1)}:
        return MIRROR_LR
    if cr + 1 < interior.shape[0] and marker == {(cr, cc), (cr + 1, cc)}:
        return MIRROR_UD
    raise AssertionError("invalid azure center marker")


def _build_seed_components(grid: np.ndarray, frame_mask: np.ndarray) -> list[dict[str, Any]]:
    seed_mask = (grid != BG) & (grid != AZURE) & (~frame_mask)
    seeds: list[dict[str, Any]] = []
    for coords in _components(seed_mask):
        r0, r1, c0, c1 = _bbox(coords)
        crop = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        for r, c in coords:
            crop[r - r0, c - c0] = int(grid[r, c])
        seeds.append(
            {
                "coords": coords,
                "r0": r0,
                "r1": r1,
                "c0": c0,
                "c1": c1,
                "crop": crop,
            }
        )
    return seeds


def _seed_corner(frame: dict[str, Any], seed: dict[str, Any]) -> str | None:
    top = any(r == frame["r0"] - 1 and frame["c0"] <= c <= frame["c1"] for r, c in seed["coords"])
    bottom = any(r == frame["r1"] + 1 and frame["c0"] <= c <= frame["c1"] for r, c in seed["coords"])
    left = any(c == frame["c0"] - 1 and frame["r0"] <= r <= frame["r1"] for r, c in seed["coords"])
    right = any(c == frame["c1"] + 1 and frame["r0"] <= r <= frame["r1"] for r, c in seed["coords"])

    if top and left:
        return "TL"
    if top and right:
        return "TR"
    if bottom and left:
        return "BL"
    if bottom and right:
        return "BR"

    mid_r = 0.5 * (frame["r0"] + frame["r1"])
    mid_c = 0.5 * (frame["c0"] + frame["c1"])
    seed_mid_r = 0.5 * (seed["r0"] + seed["r1"])
    seed_mid_c = 0.5 * (seed["c0"] + seed["c1"])

    if top:
        return "TL" if seed_mid_c <= mid_c else "TR"
    if bottom:
        return "BL" if seed_mid_c <= mid_c else "BR"
    if left:
        return "TL" if seed_mid_r <= mid_r else "BL"
    if right:
        return "TR" if seed_mid_r <= mid_r else "BR"
    return None


def _extract_local_patch(frame: dict[str, Any], seed: dict[str, Any], corner: str) -> tuple[np.ndarray, tuple[int, int]]:
    crop = seed["crop"]
    if corner == "TL":
        return crop.copy(), (0, seed["c0"] - frame["c0"])
    if corner == "TR":
        return np.fliplr(crop).copy(), (0, frame["c1"] - seed["c1"])
    if corner == "BL":
        return np.flipud(crop).copy(), (0, seed["c0"] - frame["c0"])
    if corner == "BR":
        return np.flipud(np.fliplr(crop)).copy(), (0, frame["c1"] - seed["c1"])
    raise ValueError(corner)


def _paste_local(out: np.ndarray, frame: dict[str, Any], corner: str, patch: np.ndarray, gap: tuple[int, int]) -> None:
    gr, gc = gap
    ph, pw = patch.shape
    assert gr >= 0 and gc >= 0
    assert gr + ph <= frame["inner_h"]
    assert gc + pw <= frame["inner_w"]

    inner_top = frame["r0"] + 1
    inner_bottom = frame["r1"] - 1
    inner_left = frame["c0"] + 1
    inner_right = frame["c1"] - 1

    for r, c in np.argwhere(patch != BG):
        val = int(patch[r, c])
        if corner == "TL":
            rr = inner_top + gr + int(r)
            cc = inner_left + gc + int(c)
        elif corner == "TR":
            rr = inner_top + gr + int(r)
            cc = inner_right - gc - int(c)
        elif corner == "BL":
            rr = inner_bottom - gr - int(r)
            cc = inner_left + gc + int(c)
        else:
            rr = inner_bottom - gr - int(r)
            cc = inner_right - gc - int(c)
        if out[rr, cc] not in (BG, val):
            raise AssertionError("copy overlap inside frame")
        out[rr, cc] = val


def _clear_marker(out: np.ndarray, frame: dict[str, Any]) -> None:
    interior = out[frame["r0"] + 1 : frame["r1"], frame["c0"] + 1 : frame["c1"]]
    interior[interior == AZURE] = BG


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _find_frames(g)
    assert frames

    frame_mask = np.zeros(g.shape, dtype=bool)
    for frame in frames:
        frame_mask[frame["r0"] : frame["r1"] + 1, frame["c0"]] = True
        frame_mask[frame["r0"] : frame["r1"] + 1, frame["c1"]] = True
        frame_mask[frame["r0"], frame["c0"] : frame["c1"] + 1] = True
        frame_mask[frame["r1"], frame["c0"] : frame["c1"] + 1] = True

    seeds = _build_seed_components(g, frame_mask)
    out = g.copy()
    used_seeds: set[int] = set()

    for frame in frames:
        kind = _marker_kind(g, frame)
        candidates = []
        for idx, seed in enumerate(seeds):
            if idx in used_seeds:
                continue
            corner = _seed_corner(frame, seed)
            if corner is not None:
                candidates.append((idx, seed, corner))
        assert len(candidates) == 1
        idx, seed, corner = candidates[0]
        used_seeds.add(idx)

        local_patch, gap = _extract_local_patch(frame, seed, corner)
        for r, c in seed["coords"]:
            out[r, c] = BG
        _clear_marker(out, frame)

        if kind == ROT4:
            patch = local_patch
            cur_gap = gap
            start = CORNER_ORDER.index(corner)
            for step in range(4):
                target = CORNER_ORDER[(start + step) % 4]
                _paste_local(out, frame, target, patch, cur_gap)
                patch = patch.T.copy()
                cur_gap = (cur_gap[1], cur_gap[0])
        elif kind == MIRROR_LR:
            _paste_local(out, frame, corner, local_patch, gap)
            _paste_local(out, frame, HORIZONTAL_OPPOSITE[corner], local_patch, gap)
        else:
            _paste_local(out, frame, corner, local_patch, gap)
            _paste_local(out, frame, VERTICAL_OPPOSITE[corner], local_patch, gap)

    return out


def _draw_frame(grid: np.ndarray, top: int, left: int, inner_h: int, inner_w: int, color: int) -> None:
    bottom = top + inner_h + 1
    right = left + inner_w + 1
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color


def _draw_marker(grid: np.ndarray, top: int, left: int, inner_h: int, inner_w: int, marker_kind: str) -> None:
    cr = top + 1 + (inner_h - 1) // 2
    cc = left + 1 + (inner_w - 1) // 2
    grid[cr, cc] = AZURE
    if marker_kind == MIRROR_LR:
        grid[cr, cc + 1] = AZURE
    elif marker_kind == MIRROR_UD:
        grid[cr + 1, cc] = AZURE


def _generate_connected_mask(max_h: int, max_w: int, max_area: int) -> np.ndarray | None:
    if max_h <= 0 or max_w <= 0 or max_area <= 0:
        return None
    for _ in range(80):
        h = random.randint(1, max_h)
        w = random.randint(1, max_w)
        area = random.randint(1, min(max_area, h * w))
        start = (random.randrange(h), random.randrange(w))
        pts = {start}
        frontier = {start}
        while len(pts) < area:
            cand = []
            for r, c in frontier:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in pts:
                        cand.append((nr, nc))
            if not cand:
                break
            cell = random.choice(cand)
            pts.add(cell)
            frontier.add(cell)
        if len(pts) != area:
            continue
        rows = [r for r, _ in pts]
        cols = [c for _, c in pts]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in pts:
            mask[r - r0, c - c0] = True
        return mask
    return None


def _color_patch(mask: np.ndarray, forbidden: set[int]) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    cells = list(map(tuple, np.argwhere(mask).tolist()))
    palette = [c for c in FRAME_COLORS if c not in forbidden]
    color_count = random.randint(1, min(3, len(cells), len(palette)))
    colors = random.sample(palette, color_count)
    mode = random.choice(("random", "rows", "cols", "checker"))
    random.shuffle(cells)
    for idx, (r, c) in enumerate(cells[:color_count]):
        patch[r, c] = colors[idx]
    for r, c in cells[color_count:]:
        if mode == "rows":
            val = colors[r % color_count]
        elif mode == "cols":
            val = colors[c % color_count]
        elif mode == "checker":
            val = colors[(r + c) % color_count]
        else:
            val = random.choice(colors)
        patch[r, c] = val
    return patch


def _sample_patch(inner_h: int, inner_w: int, marker_kind: str, frame_color: int) -> tuple[np.ndarray, tuple[int, int]] | None:
    max_area = max(1, (inner_h * inner_w) // 4)
    max_dim = min(inner_h, inner_w, 4)
    if max_dim <= 0:
        return None
    for _ in range(100):
        if marker_kind == ROT4:
            mask = _generate_connected_mask(max_dim, max_dim, min(max_area, max_dim * max_dim))
        else:
            mask = _generate_connected_mask(min(inner_h, 4), min(inner_w, 4), min(max_area, 8))
        if mask is None:
            continue
        ph, pw = mask.shape
        area = int(mask.sum())
        if area <= 0 or area > max_area:
            continue
        if marker_kind == ROT4:
            if ph > min(inner_h, inner_w):
                continue
            gc_max = min(inner_w - pw, inner_h - pw)
        else:
            if ph > inner_h:
                continue
            gc_max = inner_w - pw
        if gc_max < 0:
            continue
        gap = (0, random.randint(0, gc_max))
        patch = _color_patch(mask, {BG, AZURE, frame_color})
        return patch, gap
    return None


def _std_from_local(local_patch: np.ndarray, corner: str) -> np.ndarray:
    if corner == "TL":
        return local_patch.copy()
    if corner == "TR":
        return np.fliplr(local_patch).copy()
    if corner == "BL":
        return np.flipud(local_patch).copy()
    if corner == "BR":
        return np.flipud(np.fliplr(local_patch)).copy()
    raise ValueError(corner)


def _sample_frame_spec(frame_color: int) -> dict[str, Any] | None:
    marker_kind = random.choice(MARKER_TYPES)
    corner = random.choice(("TL", "TR", "BL", "BR"))
    for _ in range(120):
        inner_h = random.randint(5, 9)
        inner_w = random.randint(5, 9)
        if marker_kind == MIRROR_LR and inner_w < 2:
            continue
        if marker_kind == MIRROR_UD and inner_h < 2:
            continue
        patch_info = _sample_patch(inner_h, inner_w, marker_kind, frame_color)
        if patch_info is None:
            continue
        patch_local, gap = patch_info
        ph, pw = patch_local.shape
        if marker_kind == ROT4:
            if gap[1] + pw > min(inner_h, inner_w):
                continue
            if ph > min(inner_h, inner_w):
                continue
        box_h = inner_h + 2 + ph
        box_w = inner_w + 2
        return {
            "frame_color": frame_color,
            "marker_kind": marker_kind,
            "corner": corner,
            "inner_h": inner_h,
            "inner_w": inner_w,
            "patch_local": patch_local,
            "gap": gap,
            "box_h": box_h,
            "box_w": box_w,
        }
    return None


def _place_specs(grid_h: int, grid_w: int, specs: list[dict[str, Any]]) -> list[tuple[int, int]] | None:
    occ = np.zeros((grid_h, grid_w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(specs)
    order = sorted(range(len(specs)), key=lambda i: specs[i]["box_h"] * specs[i]["box_w"], reverse=True)
    for idx in order:
        spec = specs[idx]
        placed = False
        for _ in range(200):
            top = random.randint(0, grid_h - spec["box_h"])
            left = random.randint(0, grid_w - spec["box_w"])
            r0 = max(0, top - 1)
            r1 = min(grid_h, top + spec["box_h"] + 1)
            c0 = max(0, left - 1)
            c1 = min(grid_w, left + spec["box_w"] + 1)
            if occ[r0:r1, c0:c1].any():
                continue
            occ[top : top + spec["box_h"], left : left + spec["box_w"]] = True
            placements[idx] = (top, left)
            placed = True
            break
        if not placed:
            return None
    return [(int(r), int(c)) for r, c in placements if r is not None and c is not None]


def _render_input(grid_h: int, grid_w: int, specs: list[dict[str, Any]], placements: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for spec, (box_top, box_left) in zip(specs, placements):
        patch_local = spec["patch_local"]
        ph, pw = patch_local.shape
        corner = spec["corner"]
        gap = spec["gap"][1]
        frame_top = box_top + (ph if corner in ("TL", "TR") else 0)
        frame_left = box_left

        _draw_frame(grid, frame_top, frame_left, spec["inner_h"], spec["inner_w"], spec["frame_color"])
        _draw_marker(grid, frame_top, frame_left, spec["inner_h"], spec["inner_w"], spec["marker_kind"])

        patch_std = _std_from_local(patch_local, corner)
        seed_top = frame_top - ph if corner in ("TL", "TR") else frame_top + spec["inner_h"] + 2
        if corner in ("TL", "BL"):
            seed_left = frame_left + gap
        else:
            seed_left = frame_left + spec["inner_w"] + 2 - pw - gap
        for r, c in np.argwhere(patch_std != BG):
            grid[seed_top + int(r), seed_left + int(c)] = int(patch_std[r, c])
    return grid


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    grid_w = random.randint(*GRID_W_RANGE)
    frame_count = random.choice(FRAME_COUNT_CHOICES)
    if frame_count > len(FRAME_COLORS):
        return None

    frame_colors = random.sample(FRAME_COLORS, frame_count)
    specs = []
    for frame_color in frame_colors:
        spec = _sample_frame_spec(frame_color)
        if spec is None:
            return None
        specs.append(spec)

    placements = _place_specs(grid_h, grid_w, specs)
    if placements is None:
        return None

    inp = _render_input(grid_h, grid_w, specs, placements)
    try:
        out = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
