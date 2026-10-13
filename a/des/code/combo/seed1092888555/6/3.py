import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            sprite = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            for rr, cc in coords:
                sprite[rr - r0, cc - c0] = color
            components.append(
                {
                    "color": color,
                    "area": len(coords),
                    "bbox": (r0, c0, r1, c1),
                    "sprite": sprite,
                }
            )
    return components


def _normalize_sprite(sprite: np.ndarray, bg: int) -> np.ndarray:
    h, w = sprite.shape
    if h > w:
        return np.rot90(sprite, -1)
    return sprite.copy()


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate all objects to a horizontal long-side orientation, then pack by area."""
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    components = _find_components(g, bg)
    assert components

    packed = []
    for comp in components:
        sprite = _normalize_sprite(comp["sprite"], bg)
        packed.append((int(comp["area"]), sprite))

    packed.sort(key=lambda item: item[0])
    out_h = max(sprite.shape[0] for _, sprite in packed)
    out_w = sum(sprite.shape[1] for _, sprite in packed) + len(packed) - 1
    out = np.full((out_h, out_w), bg, dtype=int)

    col = 0
    for _area, sprite in packed:
        sh, sw = sprite.shape
        out[out_h - sh :, col : col + sw][sprite != bg] = sprite[sprite != bg]
        col += sw + 1
    return out


_RNG = np.random.default_rng()
_BG = 0
_COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
_MAX_ATTEMPTS = 400


def _rectangle_with_notch(height: int, width: int, color: int, notch_mode: str) -> np.ndarray:
    arr = np.full((height, width), color, dtype=int)
    if notch_mode == "none":
        return arr

    if notch_mode == "corner":
        corners = [(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)]
        rr, cc = corners[int(_RNG.integers(len(corners)))]
        arr[rr, cc] = _BG
        return arr

    if notch_mode == "edge2":
        if height >= width:
            col = int(_RNG.integers(width))
            start = int(_RNG.integers(height - 1))
            arr[start : start + 2, col] = _BG
        else:
            row = int(_RNG.integers(height))
            start = int(_RNG.integers(width - 1))
            arr[row, start : start + 2] = _BG
        return arr

    raise ValueError(notch_mode)


def _sample_object(area_used: set[int], force_vertical: bool) -> tuple[np.ndarray, int] | None:
    for _ in range(80):
        if force_vertical:
            height = int(_RNG.integers(4, 8))
            width = int(_RNG.integers(2, min(height, 5)))
        else:
            width = int(_RNG.integers(4, 8))
            height = int(_RNG.integers(2, min(width, 5)))

        if height == width:
            continue

        notch_choices = ["none", "corner"]
        if max(height, width) >= 4:
            notch_choices.append("edge2")
        notch_mode = notch_choices[int(_RNG.integers(len(notch_choices)))]
        sprite = _rectangle_with_notch(height, width, color=1, notch_mode=notch_mode)

        if not np.any(sprite == 1):
            continue
        if not np.any(sprite[0] == 1) or not np.any(sprite[-1] == 1):
            continue
        if not np.any(sprite[:, 0] == 1) or not np.any(sprite[:, -1] == 1):
            continue

        area = int(np.count_nonzero(sprite == 1))
        if area in area_used:
            continue
        return sprite, area
    return None


def _can_place(occupied: np.ndarray, top: int, left: int, sprite: np.ndarray) -> bool:
    h, w = sprite.shape
    H, W = occupied.shape
    if top < 0 or left < 0 or top + h > H or left + w > W:
        return False
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(H, top + h + 1)
    c1 = min(W, left + w + 1)
    if occupied[r0:r1, c0:c1].any():
        return False
    return True


def _place_sprite(grid: np.ndarray, occupied: np.ndarray, sprite: np.ndarray, color: int, top: int, left: int) -> None:
    h, w = sprite.shape
    mask = sprite == 1
    patch = grid[top : top + h, left : left + w]
    patch[mask] = color
    occupied[top : top + h, left : left + w] |= mask


def _gen_attempt() -> np.ndarray | None:
    obj_count = int(_RNG.integers(3, 6))
    vertical_count = obj_count // 2
    horizontal_count = obj_count - vertical_count
    orientations = [True] * vertical_count + [False] * horizontal_count
    _RNG.shuffle(orientations)
    if all(orientations) or not any(orientations):
        return None

    area_used: set[int] = set()
    objects: list[tuple[np.ndarray, int, bool]] = []
    for is_vertical in orientations:
        sampled = _sample_object(area_used, force_vertical=is_vertical)
        if sampled is None:
            return None
        sprite, area = sampled
        area_used.add(area)
        objects.append((sprite, area, is_vertical))

    max_h = max(sprite.shape[0] for sprite, _, _ in objects)
    max_w = max(sprite.shape[1] for sprite, _, _ in objects)
    min_h = max(14, max_h + 5)
    max_h_bound = max(min_h + 1, 23)
    min_w = max(14, obj_count * (max_w + 2))
    max_w_bound = max(min_w + 1, 29)
    grid_h = int(_RNG.integers(min_h, max_h_bound))
    grid_w = int(_RNG.integers(min_w, max_w_bound))
    grid = np.full((grid_h, grid_w), _BG, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    colors = list(_RNG.choice(_COLORS, size=obj_count, replace=False))
    order = list(_RNG.permutation(obj_count))
    for idx in order:
        sprite, _area, _is_vertical = objects[idx]
        sh, sw = sprite.shape
        placed = False
        for _ in range(120):
            top = int(_RNG.integers(0, grid_h - sh + 1))
            left = int(_RNG.integers(0, grid_w - sw + 1))
            if _can_place(occupied, top, left, sprite):
                _place_sprite(grid, occupied, sprite, colors[idx], top, left)
                placed = True
                break
        if not placed:
            return None

    comps = _find_components(grid, _BG)
    if len(comps) != obj_count:
        return None
    areas = sorted(int(comp["area"]) for comp in comps)
    if len(areas) != len(set(areas)):
        return None
    has_vertical = any(comp["sprite"].shape[0] > comp["sprite"].shape[1] for comp in comps)
    has_horizontal = any(comp["sprite"].shape[0] < comp["sprite"].shape[1] for comp in comps)
    if not (has_vertical and has_horizontal):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
