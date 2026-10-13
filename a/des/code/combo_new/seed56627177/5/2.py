import numpy as np
from collections import defaultdict


BG_COLOR = 0
CONTAINER_COLOR = 5
PIECE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
SIDES = ("up", "right", "down", "left")
PLACE_ROT_K = {"up": 0, "right": 3, "down": 2, "left": 1}
NORMALIZE_ROT_K = {"up": 0, "right": 1, "down": 2, "left": 3}
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_RNG = np.random.default_rng()

GEN_RETRIES = 300
GRID_MIN = 18
GRID_MAX = 30
CONTAINER_COUNT_CHOICES = [2, 3]
BASE_SHAPE_RETRIES = 80
PLACE_RETRIES = 250


def _most_common_color(grid: np.ndarray) -> int:
    return int(np.bincount(grid.ravel(), minlength=10).argmax())


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _crop_bool(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    return int(mask.shape[0]), int(mask.shape[1]), mask.tobytes()


def _paste(out: np.ndarray, patch: np.ndarray, top: int, left: int, mask: np.ndarray | None = None) -> None:
    h, w = patch.shape
    if mask is None:
        out[top : top + h, left : left + w] = patch
        return
    view = out[top : top + h, left : left + w]
    view[mask] = patch[mask]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _piece_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    pieces: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (bg, CONTAINER_COLOR) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not bool(seen[nr, nc]) and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords_arr = np.array(coords, dtype=int)
            r0, c0, r1, c1 = _bbox_from_coords(coords_arr)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[coords_arr[:, 0] - r0, coords_arr[:, 1] - c0] = True
            patch = np.full(mask.shape, bg, dtype=int)
            patch[mask] = color
            pieces.append(
                {
                    "coords": coords_arr,
                    "color": color,
                    "mask": mask,
                    "patch": patch,
                    "key": _shape_key(mask),
                }
            )
    return pieces


def _container_specs(grid: np.ndarray, bg: int) -> list[dict]:
    mask = grid == CONTAINER_COLOR
    containers: list[dict] = []
    for comp in _components(mask):
        r0, c0, r1, c1 = _bbox_from_coords(comp)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        bg_mask = patch == bg
        cavity_parts = []
        for part in _components(bg_mask):
            rows = part[:, 0]
            cols = part[:, 1]
            if np.any(rows == 0) or np.any(rows == bg_mask.shape[0] - 1) or np.any(cols == 0) or np.any(cols == bg_mask.shape[1] - 1):
                cavity_parts.append(part)
        if not cavity_parts:
            continue
        cavity = np.concatenate(cavity_parts, axis=0)
        touched = []
        if np.any(cavity[:, 0] == 0):
            touched.append("up")
        if np.any(cavity[:, 0] == bg_mask.shape[0] - 1):
            touched.append("down")
        if np.any(cavity[:, 1] == 0):
            touched.append("left")
        if np.any(cavity[:, 1] == bg_mask.shape[1] - 1):
            touched.append("right")
        if len(touched) != 1:
            continue
        side = touched[0]
        cr0, cc0, cr1, cc1 = _bbox_from_coords(cavity)
        cavity_mask = np.zeros((cr1 - cr0 + 1, cc1 - cc0 + 1), dtype=bool)
        cavity_mask[cavity[:, 0] - cr0, cavity[:, 1] - cc0] = True
        normalized = np.rot90(cavity_mask, NORMALIZE_ROT_K[side])
        containers.append(
            {
                "side": side,
                "paste_r": int(r0 + cr0),
                "paste_c": int(c0 + cc0),
                "cavity_mask": cavity_mask,
                "key": _shape_key(normalized),
            }
        )
    containers.sort(key=lambda d: (d["paste_r"], d["paste_c"]))
    return containers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each scattered colored fragment into the matching grey container cavity.

    Input structure:
    - The background is black.
    - Several large grey container components each occupy a solid outer rectangle minus one open cavity.
    - Each cavity touches exactly one side of its container bbox, so its opening faces up, right, down, or left.
    - The remaining foreground objects are monochrome 4-connected colored fragments, one per container.

    Rule:
    - For every grey container, extract the cavity shape and read its opening direction from the side it touches.
    - Rotate that cavity back to the canonical "opening up" orientation and match it with the unique unrotated fragment of the same shape.
    - Erase every fragment from its original location.
    - Rotate the matched fragment by the container direction and place it exactly into the cavity, preserving its color.

    Key insight:
    - Shape matching and rotation are separate: the cavity shape identifies which fragment belongs to a container, while the cavity opening alone determines the 90-degree rotation to apply.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    out = g.copy()

    pieces = _piece_components(g, bg)
    containers = _container_specs(g, bg)
    piece_map: dict[tuple[int, int, bytes], list[dict]] = defaultdict(list)
    for piece in pieces:
        piece_map[piece["key"]].append(piece)
        out[piece["coords"][:, 0], piece["coords"][:, 1]] = bg

    for container in containers:
        candidates = piece_map.get(container["key"], [])
        assert candidates, "every cavity must have a matching fragment"
        piece = candidates.pop()
        rot_k = PLACE_ROT_K[container["side"]]
        rot_patch = np.rot90(piece["patch"], rot_k)
        rot_mask = np.rot90(piece["mask"], rot_k)
        assert rot_mask.shape == container["cavity_mask"].shape
        assert np.array_equal(rot_mask, container["cavity_mask"])
        _paste(out, rot_patch, container["paste_r"], container["paste_c"], rot_mask)

    return out


def _random_base_mask() -> np.ndarray | None:
    for _ in range(BASE_SHAPE_RETRIES):
        h = int(_RNG.integers(3, 7))
        w = int(_RNG.integers(3, 7))
        open_w = int(_RNG.integers(1, min(3, w) + 1))
        start = int(_RNG.integers(0, w - open_w + 1))
        mask = np.zeros((h, w), dtype=bool)
        mask[0, start : start + open_w] = True

        min_area = max(open_w + 2, 4)
        max_area = h * w - 1
        if min_area > max_area:
            continue
        target_area = int(_RNG.integers(min_area, max_area + 1))

        while int(mask.sum()) < target_area:
            candidates = []
            for rr, cc in np.argwhere(mask):
                for dr, dc in DIR4:
                    nr, nc = int(rr + dr), int(cc + dc)
                    if not (0 <= nr < h and 0 <= nc < w) or bool(mask[nr, nc]):
                        continue
                    if nr == 0 and not (start <= nc < start + open_w):
                        continue
                    candidates.append((nr, nc))
            if not candidates:
                break
            weights = np.array([3 + nr for nr, _ in candidates], dtype=float)
            weights /= weights.sum()
            idx = int(_RNG.choice(len(candidates), p=weights))
            nr, nc = candidates[idx]
            mask[nr, nc] = True

        mask = _crop_bool(mask)
        top_cols = np.flatnonzero(mask[0])
        if top_cols.size == 0:
            continue
        if int(top_cols[-1] - top_cols[0] + 1) != int(top_cols.size):
            continue
        if int(mask.sum()) == int(mask.size):
            continue
        if not bool(mask[1:].any()):
            continue
        return mask
    return None


def _build_container(base_mask: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    h, w = base_mask.shape
    left_margin = int(_RNG.integers(1, 4))
    right_margin = int(_RNG.integers(1, 4))
    bottom_margin = int(_RNG.integers(1, 4))

    patch = np.full((h + bottom_margin, left_margin + w + right_margin), CONTAINER_COLOR, dtype=int)
    cavity = np.zeros_like(patch, dtype=bool)
    cavity[:h, left_margin : left_margin + w] = base_mask
    patch[cavity] = BG_COLOR

    rot_k = PLACE_ROT_K[side]
    return np.rot90(patch, rot_k), np.rot90(cavity, rot_k)


def _reserve_box(reserved: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> None:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(reserved.shape[0], top + h + pad)
    c1 = min(reserved.shape[1], left + w + pad)
    reserved[r0:r1, c0:c1] = True


def _try_place(reserved: np.ndarray, h: int, w: int) -> tuple[int, int] | None:
    max_r = reserved.shape[0] - h
    max_c = reserved.shape[1] - w
    if max_r < 0 or max_c < 0:
        return None
    for _ in range(PLACE_RETRIES):
        top = int(_RNG.integers(0, max_r + 1))
        left = int(_RNG.integers(0, max_c + 1))
        if reserved[top : top + h, left : left + w].any():
            continue
        _reserve_box(reserved, top, left, h, w)
        return top, left
    return None


def _gen_attempt() -> np.ndarray | None:
    count = int(_RNG.choice(CONTAINER_COUNT_CHOICES))
    colors = [int(x) for x in _RNG.choice(np.array(PIECE_COLORS), size=count, replace=False)]
    sides = [str(_RNG.choice(SIDES)) for _ in range(count)]

    base_masks: list[np.ndarray] = []
    used_keys: set[tuple[int, int, bytes]] = set()
    for _ in range(count):
        mask = _random_base_mask()
        if mask is None:
            return None
        for _retry in range(BASE_SHAPE_RETRIES):
            key = _shape_key(mask)
            if key not in used_keys:
                used_keys.add(key)
                base_masks.append(mask)
                break
            mask = _random_base_mask()
            if mask is None:
                return None
        else:
            return None

    container_patches: list[np.ndarray] = []
    cavity_masks: list[np.ndarray] = []
    piece_patches: list[np.ndarray] = []
    for mask, side, color in zip(base_masks, sides, colors):
        container_patch, cavity_mask = _build_container(mask, side)
        piece_patch = np.full(mask.shape, BG_COLOR, dtype=int)
        piece_patch[mask] = color
        container_patches.append(container_patch)
        cavity_masks.append(cavity_mask)
        piece_patches.append(piece_patch)

    tallest = max(max(p.shape[0], q.shape[0]) for p, q in zip(container_patches, piece_patches))
    widest = max(max(p.shape[1], q.shape[1]) for p, q in zip(container_patches, piece_patches))
    h_low = max(GRID_MIN, tallest + 2)
    w_low = max(GRID_MIN, widest + 2)
    if h_low > GRID_MAX or w_low > GRID_MAX:
        return None

    h = int(_RNG.integers(h_low, GRID_MAX + 1))
    w = int(_RNG.integers(w_low, GRID_MAX + 1))
    grid = np.full((h, w), BG_COLOR, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    container_pos: list[tuple[int, int] | None] = [None] * count
    for idx in sorted(range(count), key=lambda i: int(container_patches[i].size), reverse=True):
        patch = container_patches[idx]
        pos = _try_place(reserved, int(patch.shape[0]), int(patch.shape[1]))
        if pos is None:
            return None
        container_pos[idx] = pos
        _paste(grid, patch, pos[0], pos[1])

    piece_pos: list[tuple[int, int] | None] = [None] * count
    for idx in sorted(range(count), key=lambda i: int(piece_patches[i].size), reverse=True):
        patch = piece_patches[idx]
        pos = _try_place(reserved, int(patch.shape[0]), int(patch.shape[1]))
        if pos is None:
            return None
        piece_pos[idx] = pos
        _paste(grid, patch, pos[0], pos[1], patch != BG_COLOR)

    expected = grid.copy()
    for idx, patch in enumerate(piece_patches):
        top, left = piece_pos[idx]
        expected[top : top + patch.shape[0], left : left + patch.shape[1]][patch != BG_COLOR] = BG_COLOR
    for idx, cavity in enumerate(cavity_masks):
        top, left = container_pos[idx]
        expected[top : top + cavity.shape[0], left : left + cavity.shape[1]][cavity] = colors[idx]

    if len(_container_specs(grid, BG_COLOR)) != count:
        return None
    if len(_piece_components(grid, BG_COLOR)) != count:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid container-fragment sample")
