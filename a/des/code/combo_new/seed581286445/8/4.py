from __future__ import annotations

import itertools

import numpy as np


BG = 8
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 9)
GEN_ATTEMPTS = 300
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
EDGE_TO_ROT = {"top": 0, "right": 1, "bottom": 2, "left": 3}
EDGES = tuple(EDGE_TO_ROT)

SHAPES: dict[str, np.ndarray] = {
    "domino": np.array([[1, 1], [0, 0]], dtype=bool),
    "l3": np.array([[1, 1], [1, 0]], dtype=bool),
    "i3": np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=bool),
    "t4": np.array([[1, 1, 1], [0, 1, 0], [0, 0, 0]], dtype=bool),
    "l4": np.array([[1, 1, 1], [1, 0, 0], [0, 0, 0]], dtype=bool),
    "z4": np.array([[1, 1, 0], [0, 1, 1], [0, 0, 0]], dtype=bool),
}


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            pts = np.asarray(coords, dtype=int)
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            comps.append(
                {
                    "coords": pts,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(coords)),
                }
            )
    return comps


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(grid == color):
            comp["color"] = color
            comps.append(comp)
    return comps


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    assert pts.size > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _canonical_rot_key(mask: np.ndarray) -> tuple[bytes, tuple[int, int]]:
    tight = _crop_mask(mask)
    best: tuple[bytes, tuple[int, int]] | None = None
    for k in range(4):
        rot = np.rot90(tight, k)
        key = (rot.tobytes(), tuple(map(int, rot.shape)))
        if best is None or key < best:
            best = key
    assert best is not None
    return best


def _paint_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    mask = patch != BG
    view = grid[r0 : r0 + h, c0 : c0 + w]
    view[mask] = patch[mask]


def _shape_signature_from_component(comp: dict) -> tuple[bytes, tuple[int, int]]:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp["coords"]:
        mask[int(r) - r0, int(c) - c0] = True
    return _canonical_rot_key(mask)


def _container_info(sub: np.ndarray, bg: int) -> dict:
    h, w = sub.shape
    bg_comps = _components(sub == bg)
    notch_comp: dict | None = None
    hole_count = 0

    for comp in bg_comps:
        r0, c0, r1, c1 = comp["bbox"]
        touches_border = r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1
        if touches_border:
            assert notch_comp is None, "expected one border-connected notch"
            notch_comp = comp
        else:
            assert comp["area"] == 1, "interior bg regions must be 1-pixel holes"
            hole_count += 1

    assert notch_comp is not None, "container must expose one notch"

    coords = notch_comp["coords"]
    touches = []
    if np.any(coords[:, 0] == 0):
        touches.append("top")
    if np.any(coords[:, 0] == h - 1):
        touches.append("bottom")
    if np.any(coords[:, 1] == 0):
        touches.append("left")
    if np.any(coords[:, 1] == w - 1):
        touches.append("right")
    assert len(touches) == 1, "notch must touch exactly one container edge"
    edge = touches[0]

    nr0, nc0, nr1, nc1 = notch_comp["bbox"]
    size = max(int(nr1 - nr0 + 1), int(nc1 - nc0 + 1))
    if edge == "top":
        box_r0, box_c0 = 0, int(nc0)
    elif edge == "bottom":
        box_r0, box_c0 = h - size, int(nc0)
    elif edge == "left":
        box_r0, box_c0 = int(nr0), 0
    else:
        box_r0, box_c0 = int(nr0), w - size

    assert 0 <= box_r0 <= h - size
    assert 0 <= box_c0 <= w - size

    slot_mask = np.zeros((size, size), dtype=bool)
    for r, c in coords:
        slot_mask[int(r) - box_r0, int(c) - box_c0] = True

    return {
        "edge": edge,
        "hole_count": int(hole_count),
        "size": int(size),
        "box_r0": int(box_r0),
        "box_c0": int(box_c0),
        "slot_mask": slot_mask,
        "shape_key": _canonical_rot_key(slot_mask),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each scattered key to its container notch, then rotate the key again by the container's hole count before insertion.

    Input structure:
    - The background is azure.
    - Large monochrome rectangular containers have one edge-connected notch cut out of them and may also contain 0-3 enclosed 1-pixel azure holes.
    - Small monochrome key components of area 2-4 are scattered elsewhere, each with a different color from its matching container.
    - Every container's notch shape matches exactly one key up to rotation.

    Rule:
    - Separate large containers from small keys and, for each container, extract the unique border-connected notch and count the enclosed 1-pixel holes.
    - Match each key to the container whose notch has the same shape under rotation.
    - Take the notch-aligned placement as the base fit, then rotate that placement clockwise by N x 90 degrees, where N is the container's hole count.
    - Erase the key from its original location and draw the rotated key in the container's notch zone using the key's own color.

    Key insight:
    - The task requires both shape-complement matching and a second rotation cue: the container's enclosed hole count changes the final orientation after the key-container pairing is already known.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    comps = _monochrome_components(g, bg)

    containers = []
    keys = []
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h >= 4 and w >= 4:
            containers.append(comp)
        else:
            keys.append(comp)

    assert containers, "expected at least one container"
    assert len(containers) == len(keys), "containers and keys must pair one-to-one"

    key_infos = []
    for comp in keys:
        key_infos.append(
            {
                "comp": comp,
                "color": int(comp["color"]),
                "shape_key": _shape_signature_from_component(comp),
                "used": False,
            }
        )

    out = g.copy()
    for info in key_infos:
        for r, c in info["comp"]["coords"]:
            out[int(r), int(c)] = bg

    for comp in containers:
        r0, c0, r1, c1 = comp["bbox"]
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        info = _container_info(sub, bg)
        matches = [item for item in key_infos if not item["used"] and item["shape_key"] == info["shape_key"]]
        assert len(matches) == 1, "each container must have a unique matching key"
        key = matches[0]
        key["used"] = True

        size = int(info["size"])
        box_r0 = int(info["box_r0"])
        box_c0 = int(info["box_c0"])
        container_color = int(comp["color"])
        key_color = int(key["color"])

        out[r0 + box_r0 : r0 + box_r0 + size, c0 + box_c0 : c0 + box_c0 + size] = container_color
        final_mask = np.rot90(info["slot_mask"], -int(info["hole_count"]) % 4)
        target = out[r0 + box_r0 : r0 + box_r0 + size, c0 + box_c0 : c0 + box_c0 + size]
        target[final_mask] = key_color

    return out


def _rot_cw(mask: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(mask, -int(k))


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(int(lo), int(hi) + 1))


def _choose_isolated_cells(
    candidates: list[tuple[int, int]],
    count: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    if count == 0:
        return []
    order = candidates[:]
    rng.shuffle(order)

    def backtrack(index: int, chosen: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(chosen) == count:
            return chosen[:]
        if index >= len(order):
            return None
        for i in range(index, len(order)):
            cell = order[i]
            if any(abs(cell[0] - rr) + abs(cell[1] - cc) == 1 for rr, cc in chosen):
                continue
            chosen.append(cell)
            result = backtrack(i + 1, chosen)
            if result is not None:
                return result
            chosen.pop()
        return None

    return backtrack(0, [])


def _make_container_patch(
    rng: np.random.Generator,
    *,
    shape_name: str,
    edge: str,
    color: int,
    hole_count: int,
) -> np.ndarray | None:
    top_mask = SHAPES[shape_name]
    slot_mask = _rot_cw(top_mask, EDGE_TO_ROT[edge])
    size = int(slot_mask.shape[0])

    for _ in range(120):
        min_h = max(5, size + 2)
        min_w = max(5, size + 2)
        max_side = 6
        if hole_count >= 3 or size == 3:
            max_side = 7
        h = _randint(rng, min_h, max_side)
        w = _randint(rng, min_w, max_side)

        if edge in ("top", "bottom"):
            positions = list(range(1, w - size))
            if not positions:
                continue
            box_r0 = 0 if edge == "top" else h - size
            box_c0 = int(rng.choice(positions))
        else:
            positions = list(range(1, h - size))
            if not positions:
                continue
            box_r0 = int(rng.choice(positions))
            box_c0 = 0 if edge == "left" else w - size

        patch = np.full((h, w), int(color), dtype=int)
        for rr, cc in np.argwhere(slot_mask):
            patch[box_r0 + int(rr), box_c0 + int(cc)] = BG

        candidates: list[tuple[int, int]] = []
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if box_r0 <= r < box_r0 + size and box_c0 <= c < box_c0 + size:
                    continue
                if patch[r, c] == BG:
                    continue
                if any(patch[r + dr, c + dc] == BG for dr, dc in DIR4):
                    continue
                candidates.append((int(r), int(c)))

        holes = _choose_isolated_cells(candidates, hole_count, rng)
        if holes is None:
            continue

        for r, c in holes:
            patch[int(r), int(c)] = BG

        try:
            info = _container_info(patch, BG)
        except AssertionError:
            continue
        if info["hole_count"] != hole_count:
            continue
        if info["shape_key"] != _canonical_rot_key(slot_mask):
            continue
        return patch

    return None


def _make_key_patch(rng: np.random.Generator, *, shape_name: str, color: int) -> np.ndarray:
    rot = _randint(rng, 0, 3)
    mask = _rot_cw(SHAPES[shape_name], rot)
    tight = _crop_mask(mask)
    patch = np.full(tight.shape, BG, dtype=int)
    patch[tight] = int(color)
    return patch


def _place_patch(
    grid: np.ndarray,
    occupied: np.ndarray,
    patch: np.ndarray,
    rng: np.random.Generator,
    *,
    padding: int = 1,
) -> bool:
    h, w = patch.shape
    H, W = grid.shape
    rows = list(range(0, H - h + 1))
    cols = list(range(0, W - w + 1))
    rng.shuffle(rows)
    rng.shuffle(cols)
    for r0 in rows:
        for c0 in cols:
            rr0 = max(0, r0 - padding)
            rr1 = min(H, r0 + h + padding)
            cc0 = max(0, c0 - padding)
            cc1 = min(W, c0 + w + padding)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            _paint_patch(grid, patch, int(r0), int(c0))
            occupied[r0 : r0 + h, c0 : c0 + w] = True
            return True
    return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n = _randint(rng, 1, 3)
    shape_names = list(SHAPES)
    rng.shuffle(shape_names)
    chosen_shapes = shape_names[:n]

    colors = list(COLOR_POOL)
    rng.shuffle(colors)
    container_colors = colors[:n]
    key_colors = colors[n : 2 * n]

    patches: list[np.ndarray] = []
    for idx, shape_name in enumerate(chosen_shapes):
        edge = EDGES[_randint(rng, 0, 3)]
        hole_count = _randint(rng, 0, 3)
        container = _make_container_patch(
            rng,
            shape_name=shape_name,
            edge=edge,
            color=int(container_colors[idx]),
            hole_count=hole_count,
        )
        if container is None:
            return None
        key = _make_key_patch(rng, shape_name=shape_name, color=int(key_colors[idx]))
        patches.append(container)
        patches.append(key)

    patches.sort(key=lambda arr: (arr.shape[0] * arr.shape[1], arr.shape[0], arr.shape[1]), reverse=True)
    min_side = {1: 12, 2: 18, 3: 22}[n]
    H = _randint(rng, min_side, 30)
    W = _randint(rng, min_side, 30)
    grid = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    for patch in patches:
        if not _place_patch(grid, occupied, patch, rng, padding=1):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
