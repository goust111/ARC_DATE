from __future__ import annotations

import math

import numpy as np


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray, bg: int = 0) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )

    return comps


def _component_patch(comp: dict, bg: int = 0) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in comp["coords"]:
        patch[int(r) - r0, int(c) - c0] = int(comp["color"])
    return patch


def _is_container_component(grid: np.ndarray, comp: dict, bg: int = 0) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if int(comp["area"]) != 2 * (h + w) - 4:
        return False

    color = int(comp["color"])
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.all(patch[border] == color) and np.all(patch[~border] == bg))


def _pack_pieces(pieces: list[dict], inner_w: int) -> tuple[list[tuple[int, int]], int, bool]:
    placements: list[tuple[int, int]] = []
    cur_r = 0
    cur_c = 0
    row_h = 0
    wrapped = False

    for piece in pieces:
        ph, pw = piece["patch"].shape
        if cur_c > 0 and cur_c + pw > inner_w:
            cur_r += row_h
            cur_c = 0
            row_h = 0
            wrapped = True
        assert pw <= inner_w
        placements.append((cur_r, cur_c))
        cur_c += pw
        row_h = max(row_h, ph)

    used_h = cur_r + row_h
    return placements, used_h, wrapped


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group scattered monochrome fragments by color and pack them into same-colored framed containers.

    Rule:
    - Rectangular 1-pixel frames with black interiors are the containers.
    - Other 4-connected monochrome components are fragments.
    - For each color, sort fragments by area descending and place them inside the matching container.
    - Placement uses each fragment's bounding box, filling rows from left to right and wrapping tightly.
    - Original fragment locations are cleared to black.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = 0

    containers: dict[int, dict] = {}
    fragments: dict[int, list[dict]] = {}
    comps = _find_components(g, bg=bg)

    for comp in comps:
        color = int(comp["color"])
        if _is_container_component(g, comp, bg=bg):
            assert color not in containers
            containers[color] = comp
        else:
            item = dict(comp)
            item["patch"] = _component_patch(comp, bg=bg)
            fragments.setdefault(color, []).append(item)

    assert containers
    assert set(containers) == set(fragments)

    out = g.copy()
    for group in fragments.values():
        for piece in group:
            coords = piece["coords"]
            out[coords[:, 0], coords[:, 1]] = bg

    for color, container in containers.items():
        pieces = sorted(
            fragments[color],
            key=lambda piece: (-int(piece["area"]), int(piece["bbox"][0]), int(piece["bbox"][1])),
        )
        cr0, cc0, cr1, cc1 = container["bbox"]
        inner_r0 = cr0 + 1
        inner_c0 = cc0 + 1
        inner_h = cr1 - cr0 - 1
        inner_w = cc1 - cc0 - 1
        placements, _used_h, _wrapped = _pack_pieces(pieces, inner_w)

        for piece, (pr, pc) in zip(pieces, placements, strict=False):
            ph, pw = piece["patch"].shape
            assert pr + ph <= inner_h
            assert pc + pw <= inner_w
            region = out[inner_r0 + pr : inner_r0 + pr + ph, inner_c0 + pc : inner_c0 + pc + pw]
            mask = piece["patch"] != bg
            region[mask] = piece["patch"][mask]

    return out


_RNG = np.random.default_rng()
_COLORS = np.arange(1, 10, dtype=int)
_MAX_GENERATION_ATTEMPTS = 400


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _is_frame_mask(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.all(mask[border]) and not np.any(mask[~border]))


def _random_connected_mask(area: int) -> np.ndarray:
    dim_candidates = [
        (h, w)
        for h in range(1, 6)
        for w in range(1, 6)
        if area <= h * w <= area + 6
    ]
    assert dim_candidates

    for _ in range(200):
        h, w = dim_candidates[int(_RNG.integers(len(dim_candidates)))]
        mask = np.zeros((h, w), dtype=bool)
        start_r = int(_RNG.integers(h))
        start_c = int(_RNG.integers(w))
        mask[start_r, start_c] = True

        while int(mask.sum()) < area:
            frontier: set[tuple[int, int]] = set()
            for r, c in np.argwhere(mask):
                for nr, nc in _neighbors4(int(r), int(c)):
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.add((nr, nc))
            assert frontier
            choice = list(frontier)[int(_RNG.integers(len(frontier)))]
            mask[choice] = True

        mask = _crop_mask(mask)
        if not _is_frame_mask(mask):
            return mask

    raise RuntimeError("failed to build fragment mask")


def _generate_piece(color: int, area: int) -> dict:
    mask = _random_connected_mask(area)
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return {"color": int(color), "area": int(area), "patch": patch}


def _build_group(color: int, wrap_flag: bool) -> dict | None:
    n_pieces = int(_RNG.integers(2, 5))
    areas = _RNG.choice(np.arange(2, 10), size=n_pieces, replace=False)
    pieces = [_generate_piece(int(color), int(area)) for area in sorted(map(int, areas), reverse=True)]

    total_w = sum(piece["patch"].shape[1] for piece in pieces)
    max_w = max(piece["patch"].shape[1] for piece in pieces)
    if wrap_flag:
        if total_w - 1 < max_w:
            return None
        inner_w = int(_RNG.integers(max_w, total_w))
    else:
        inner_w = int(total_w + _RNG.integers(0, 3))

    placements, used_h, wrapped = _pack_pieces(pieces, inner_w)
    if wrap_flag != wrapped:
        return None

    inner_h = int(used_h + _RNG.integers(0, 3))
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    if outer_h > 15 or outer_w > 15:
        return None

    container_patch = np.zeros((outer_h, outer_w), dtype=int)
    container_patch[0, :] = color
    container_patch[-1, :] = color
    container_patch[:, 0] = color
    container_patch[:, -1] = color

    return {
        "color": int(color),
        "pieces": pieces,
        "placements": placements,
        "inner_h": int(inner_h),
        "inner_w": int(inner_w),
        "container_patch": container_patch,
    }


def _place_with_margin(blocked: np.ndarray, h: int, w: int) -> tuple[int, int] | None:
    H, W = blocked.shape
    candidates: list[tuple[int, int]] = []
    for r in range(H - h + 1):
        for c in range(W - w + 1):
            r0 = max(0, r - 1)
            c0 = max(0, c - 1)
            r1 = min(H, r + h + 1)
            c1 = min(W, c + w + 1)
            if not bool(blocked[r0:r1, c0:c1].any()):
                candidates.append((r, c))
    if not candidates:
        return None
    return candidates[int(_RNG.integers(len(candidates)))]


def _mark_blocked(blocked: np.ndarray, r: int, c: int, h: int, w: int) -> None:
    H, W = blocked.shape
    r0 = max(0, r - 1)
    c0 = max(0, c - 1)
    r1 = min(H, r + h + 1)
    c1 = min(W, c + w + 1)
    blocked[r0:r1, c0:c1] = True


def _approx_canvas(groups: list[dict]) -> tuple[int, int] | None:
    footprint = 0
    max_h = 0
    max_w = 0
    for group in groups:
        ch, cw = group["container_patch"].shape
        footprint += (ch + 2) * (cw + 2)
        max_h = max(max_h, ch)
        max_w = max(max_w, cw)
        for piece in group["pieces"]:
            ph, pw = piece["patch"].shape
            footprint += (ph + 2) * (pw + 2)
            max_h = max(max_h, ph)
            max_w = max(max_w, pw)

    base = int(math.ceil(math.sqrt(footprint * float(_RNG.uniform(1.8, 2.5)))))
    H = max(max_h + 2, base + int(_RNG.integers(-2, 3)))
    W = max(max_w + 2, base + int(_RNG.integers(-2, 3)))
    H = int(np.clip(H, 18, 30))
    W = int(np.clip(W, 18, 30))
    if H < max_h or W < max_w:
        return None
    return H, W


def _build_scene(groups: list[dict]) -> np.ndarray | None:
    canvas = _approx_canvas(groups)
    if canvas is None:
        return None
    H, W = canvas
    grid = np.zeros((H, W), dtype=int)
    blocked = np.zeros((H, W), dtype=bool)

    groups_order = sorted(
        groups,
        key=lambda group: group["container_patch"].shape[0] * group["container_patch"].shape[1],
        reverse=True,
    )
    for group in groups_order:
        patch = group["container_patch"]
        h, w = patch.shape
        pos = _place_with_margin(blocked, h, w)
        if pos is None:
            return None
        r, c = pos
        grid[r : r + h, c : c + w] = np.maximum(grid[r : r + h, c : c + w], patch)
        _mark_blocked(blocked, r, c, h, w)
        group["container_pos"] = (int(r), int(c))

    frag_records: list[dict] = []
    piece_order: list[tuple[dict, dict]] = []
    for group in groups:
        for piece in group["pieces"]:
            piece_order.append((group, piece))
    piece_order.sort(
        key=lambda item: (
            -(item[1]["patch"].shape[0] * item[1]["patch"].shape[1]),
            -int(item[1]["area"]),
        )
    )

    for group, piece in piece_order:
        patch = piece["patch"]
        h, w = patch.shape
        pos = _place_with_margin(blocked, h, w)
        if pos is None:
            return None
        r, c = pos
        region = grid[r : r + h, c : c + w]
        mask = patch != 0
        region[mask] = patch[mask]
        _mark_blocked(blocked, r, c, h, w)
        frag_records.append({"group": group, "piece": piece, "pos": (int(r), int(c))})

    for group in groups:
        group["fragments"] = [record for record in frag_records if record["group"] is group]

    return grid


def _expected_output(grid: np.ndarray, groups: list[dict]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for group in groups:
        for record in group["fragments"]:
            patch = record["piece"]["patch"]
            r, c = record["pos"]
            h, w = patch.shape
            mask = patch != 0
            out[r : r + h, c : c + w][mask] = 0

    for group in groups:
        cr, cc = group["container_pos"]
        for piece, (pr, pc) in zip(group["pieces"], group["placements"], strict=False):
            patch = piece["patch"]
            h, w = patch.shape
            mask = patch != 0
            out[cr + 1 + pr : cr + 1 + pr + h, cc + 1 + pc : cc + 1 + pc + w][mask] = patch[mask]

    return out


def _attempt_input() -> np.ndarray | None:
    n_colors = int(_RNG.integers(2, 5))
    colors = [int(v) for v in _RNG.choice(_COLORS, size=n_colors, replace=False)]
    wrap_flags = [i < n_colors // 2 for i in range(n_colors)]
    _RNG.shuffle(wrap_flags)

    groups: list[dict] = []
    for color, wrap_flag in zip(colors, wrap_flags, strict=False):
        group = _build_group(color, bool(wrap_flag))
        if group is None:
            return None
        groups.append(group)

    grid = _build_scene(groups)
    if grid is None:
        return None

    expected = _expected_output(grid, groups)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _attempt_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
