from __future__ import annotations

from collections import Counter

import numpy as np


BG_COLOR = 7
COLOR_POOL = tuple(c for c in range(10) if c != BG_COLOR)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RNG = np.random.default_rng()
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _component_records(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(pts, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            rel = coords - np.array([r0, c0], dtype=int)
            comp_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            comp_mask[rel[:, 0], rel[:, 1]] = True
            comps.append(
                {
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": comp_mask,
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _unique_orientations(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for flip in (False, True):
        base = np.fliplr(mask) if flip else mask
        for k in range(4):
            cur = np.rot90(base, k)
            key = (tuple(map(int, cur.shape)), cur.astype(np.uint8).tobytes())
            if key in seen:
                continue
            seen.add(key)
            variants.append(cur.copy())
    return variants


def _matching_holes(container_patch: np.ndarray, bg: int, piece_mask: np.ndarray) -> list[np.ndarray]:
    holes = _component_records(container_patch == bg)
    piece_oris = _unique_orientations(piece_mask)
    matches: list[np.ndarray] = []
    for hole in holes:
        hole_mask = hole["mask"]
        if any(ori.shape == hole_mask.shape and np.array_equal(ori, hole_mask) for ori in piece_oris):
            matches.append(hole["coords"])
    return matches


def _detect_pair_action(grid: np.ndarray, bg: int, color: int, comps: list[dict]) -> dict:
    assert len(comps) == 2
    actions: list[dict] = []
    for container, piece in ((comps[0], comps[1]), (comps[1], comps[0])):
        r0, c0, r1, c1 = container["bbox"]
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        matches = _matching_holes(patch, bg, piece["mask"])
        if len(matches) != 1:
            continue
        hole_abs = matches[0] + np.array([r0, c0], dtype=int)
        actions.append(
            {
                "color": int(color),
                "erase": piece["coords"],
                "fill": hole_abs,
                "container_score": ((r1 - r0 + 1) * (c1 - c0 + 1), int(container["area"])),
            }
        )
    assert len(actions) == 1, f"expected one complementary match for color {color}, got {len(actions)}"
    return actions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-color objects by complementary shape and move the smaller one into the larger gap.

    The background is orange. For each color that appears in exactly two 4-connected
    components, exactly one component's bbox contains a background-colored gap whose
    shape matches the other component up to rotation/reflection. Fill that gap with the
    shared color and erase the moved component's original position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _get_bg(g)
    out = g.copy()

    for color in sorted(int(c) for c in np.unique(g) if int(c) != bg):
        comps = _component_records(g == int(color))
        if len(comps) != 2:
            continue
        action = _detect_pair_action(g, int(bg), int(color), comps)
        erase = action["erase"]
        fill = action["fill"]
        out[erase[:, 0], erase[:, 1]] = int(bg)
        out[fill[:, 0], fill[:, 1]] = int(color)

    return out


def _mask_has_enclosed_hole(mask: np.ndarray) -> bool:
    for comp in _component_records(~mask):
        coords = comp["coords"]
        touches_edge = (
            np.any(coords[:, 0] == 0)
            or np.any(coords[:, 0] == mask.shape[0] - 1)
            or np.any(coords[:, 1] == 0)
            or np.any(coords[:, 1] == mask.shape[1] - 1)
        )
        if not touches_edge:
            return True
    return False


def _random_connected_mask(
    min_side: int = 2,
    max_side: int = 5,
    min_area: int = 3,
    max_area: int = 8,
    require_multiple_orientations: bool = False,
) -> np.ndarray | None:
    for _ in range(200):
        h = int(RNG.integers(min_side, max_side + 1))
        w = int(RNG.integers(min_side, max_side + 1))
        limit = min(max_area, h * w)
        if limit < min_area:
            continue
        area = int(RNG.integers(min_area, limit + 1))
        mask = np.zeros((h, w), dtype=bool)
        start = (int(RNG.integers(h)), int(RNG.integers(w)))
        mask[start] = True
        frontier: set[tuple[int, int]] = set()
        for dr, dc in DIR4:
            nr, nc = start[0] + dr, start[1] + dc
            if 0 <= nr < h and 0 <= nc < w:
                frontier.add((nr, nc))
        while int(mask.sum()) < area and frontier:
            cell = tuple(frontier.pop()) if len(frontier) == 1 else frontier.pop()
            if len(frontier) > 0:
                frontier.add(cell)
                cell = list(frontier)[int(RNG.integers(len(frontier)))]
                frontier.discard(cell)
            r, c = map(int, cell)
            if mask[r, c]:
                continue
            mask[r, c] = True
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if int(mask.sum()) != area:
            continue
        mask = _trim_mask(mask)
        if mask.shape[0] < 2 and mask.shape[1] < 2:
            continue
        if _mask_has_enclosed_hole(mask):
            continue
        if require_multiple_orientations and len(_unique_orientations(mask)) < 2:
            continue
        return mask
    return None


def _paint_mask(mask: np.ndarray, color: int, bg: int = BG_COLOR) -> np.ndarray:
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(color)
    return patch


def _build_pair(color: int) -> dict | None:
    require_transform = bool(RNG.integers(2))
    piece_base = _random_connected_mask(require_multiple_orientations=require_transform)
    if piece_base is None:
        return None
    oris = _unique_orientations(piece_base)
    small_mask = oris[int(RNG.integers(len(oris)))]
    if require_transform and len(oris) >= 2:
        alt = [ori for ori in oris if not np.array_equal(ori, small_mask)]
        hole_mask = alt[int(RNG.integers(len(alt)))]
    else:
        hole_mask = oris[int(RNG.integers(len(oris)))]

    hh, hw = hole_mask.shape
    edge_notch = bool(RNG.integers(2))
    if edge_notch:
        zero_side = int(RNG.integers(4))
        margins = [int(RNG.integers(1, 4)) for _ in range(4)]
        margins[zero_side] = 0
    else:
        margins = [int(RNG.integers(1, 4)) for _ in range(4)]
    top, bottom, left, right = margins
    host_h = top + hh + bottom
    host_w = left + hw + right
    host_mask = np.ones((host_h, host_w), dtype=bool)
    hole_slice = host_mask[top : top + hh, left : left + hw]
    hole_slice[hole_mask] = False

    comps = _component_records(host_mask)
    if len(comps) != 1:
        return None
    if int(host_mask.sum()) <= int(small_mask.sum()):
        return None

    hole_rel = np.argwhere(~host_mask)
    return {
        "color": int(color),
        "container_patch": _paint_mask(host_mask, int(color)),
        "small_patch": _paint_mask(small_mask, int(color)),
        "hole_rel": hole_rel,
        "small_rel": np.argwhere(small_mask),
        "edge_notch": edge_notch,
        "needs_transform": require_transform,
    }


def _place_patch(grid: np.ndarray, reserved: np.ndarray, patch: np.ndarray, pad: int = 1) -> tuple[int, int] | None:
    h, w = grid.shape
    ph, pw = patch.shape
    choices: list[tuple[int, int]] = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            rr0 = max(0, r0 - pad)
            rr1 = min(h, r0 + ph + pad)
            cc0 = max(0, c0 - pad)
            cc1 = min(w, c0 + pw + pad)
            if np.any(reserved[rr0:rr1, cc0:cc1]):
                continue
            choices.append((r0, c0))
    if not choices:
        return None
    r0, c0 = choices[int(RNG.integers(len(choices)))]
    mask = patch != BG_COLOR
    grid[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]
    rr0 = max(0, r0 - pad)
    rr1 = min(h, r0 + ph + pad)
    cc0 = max(0, c0 - pad)
    cc1 = min(w, c0 + pw + pad)
    reserved[rr0:rr1, cc0:cc1] = True
    return int(r0), int(c0)


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    h = int(RNG.integers(15, 25))
    w = int(RNG.integers(15, 25))
    grid = np.full((h, w), BG_COLOR, dtype=int)
    expected = grid.copy()
    reserved = np.zeros((h, w), dtype=bool)

    num_pairs = int(RNG.integers(1, 4))
    num_singletons = int(RNG.integers(0, 3))
    if num_pairs + num_singletons > len(COLOR_POOL):
        return None
    colors = list(RNG.permutation(np.array(COLOR_POOL)))[: num_pairs + num_singletons]

    pair_specs: list[dict] = []
    jobs: list[dict] = []
    for pair_id in range(num_pairs):
        spec = _build_pair(int(colors[pair_id]))
        if spec is None:
            return None
        pair_specs.append(spec)
        jobs.append(
            {
                "kind": "container",
                "pair_id": pair_id,
                "color": int(spec["color"]),
                "patch": spec["container_patch"],
                "rel": spec["hole_rel"],
            }
        )
        jobs.append(
            {
                "kind": "small",
                "pair_id": pair_id,
                "color": int(spec["color"]),
                "patch": spec["small_patch"],
                "rel": spec["small_rel"],
            }
        )

    for idx in range(num_singletons):
        color = int(colors[num_pairs + idx])
        mask = _random_connected_mask(min_side=2, max_side=4, min_area=3, max_area=6)
        if mask is None:
            return None
        jobs.append(
            {
                "kind": "singleton",
                "pair_id": -1,
                "color": color,
                "patch": _paint_mask(mask, color),
                "rel": np.argwhere(mask),
            }
        )

    order = list(RNG.permutation(len(jobs)))
    pair_meta = [{} for _ in range(num_pairs)]
    for job_index in order:
        job = jobs[int(job_index)]
        pos = _place_patch(grid, reserved, job["patch"], pad=1)
        if pos is None:
            return None
        expected[:, :] = grid
        r0, c0 = pos
        abs_coords = job["rel"] + np.array([r0, c0], dtype=int)
        if job["kind"] == "container":
            pair_meta[job["pair_id"]]["hole_abs"] = abs_coords
            pair_meta[job["pair_id"]]["color"] = int(job["color"])
        elif job["kind"] == "small":
            pair_meta[job["pair_id"]]["small_abs"] = abs_coords

    expected = grid.copy()
    for meta in pair_meta:
        if "hole_abs" not in meta or "small_abs" not in meta:
            return None
        hole_abs = meta["hole_abs"]
        small_abs = meta["small_abs"]
        color = int(meta["color"])
        expected[small_abs[:, 0], small_abs[:, 1]] = BG_COLOR
        expected[hole_abs[:, 0], hole_abs[:, 1]] = color

    if Counter(map(int, grid.ravel())).most_common(1)[0][0] != BG_COLOR:
        return None
    if np.array_equal(grid, expected):
        return None
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        attempt = _gen_attempt()
        if attempt is None:
            continue
        grid, _ = attempt
        return grid
    raise RuntimeError("failed to generate a valid complementary-pair ARC input")
