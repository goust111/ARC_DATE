from __future__ import annotations

from collections import deque
import random

import numpy as np


DIRECTIONS = ("up", "down", "left", "right")
BG = 0
COLOR_POOL = tuple(range(1, 10))
GENERATION_RETRIES = 400
PATTERN_RETRIES = 120


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(counts.argmax())])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(int(r), int(c))])
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            out.append(np.array(cells, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _u_mask(shape: tuple[int, int], direction: str) -> np.ndarray | None:
    h, w = shape
    if direction in ("up", "down"):
        if h < 3 or w < 4:
            return None
    else:
        if h < 4 or w < 3:
            return None

    mask = np.zeros(shape, dtype=bool)
    if direction == "up":
        mask[:, 0] = True
        mask[:, -1] = True
        mask[-1, :] = True
    elif direction == "down":
        mask[:, 0] = True
        mask[:, -1] = True
        mask[0, :] = True
    elif direction == "left":
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, -1] = True
    else:
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
    return mask


def _opening_endpoints(shape: tuple[int, int], direction: str) -> set[tuple[int, int]]:
    h, w = shape
    if direction == "up":
        return {(0, 0), (0, w - 1)}
    if direction == "down":
        return {(h - 1, 0), (h - 1, w - 1)}
    if direction == "left":
        return {(0, 0), (h - 1, 0)}
    return {(0, w - 1), (h - 1, w - 1)}


def _interior_rect(shape: tuple[int, int], direction: str) -> tuple[int, int, int, int]:
    h, w = shape
    if direction == "up":
        return 0, 1, h - 1, w - 2
    if direction == "down":
        return 1, 1, h - 1, w - 2
    if direction == "left":
        return 1, 0, h - 2, w - 1
    return 1, 1, h - 2, w - 1


def _component_patch(grid: np.ndarray, coords: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    rows = coords[:, 0] - r0
    cols = coords[:, 1] - c0
    patch[rows, cols] = grid[coords[:, 0], coords[:, 1]]
    return patch, (r0, c0, r1, c1)


def _classify_container(grid: np.ndarray, coords: np.ndarray, bg: int):
    patch, bbox_ = _component_patch(grid, coords, bg)
    r0, c0, r1, c1 = bbox_
    local_mask = np.zeros_like(patch, dtype=bool)
    local_mask[coords[:, 0] - r0, coords[:, 1] - c0] = True

    if np.any((grid[r0 : r1 + 1, c0 : c1 + 1] != bg) & ~local_mask):
        return None

    matches = []
    for direction in DIRECTIONS:
        target_mask = _u_mask(patch.shape, direction)
        if target_mask is None or not np.array_equal(local_mask, target_mask):
            continue
        values = patch[target_mask]
        colors, counts = np.unique(values, return_counts=True)
        if len(colors) != 2:
            continue
        major = int(colors[int(counts.argmax())])
        if major == bg or int(counts.max()) != int(target_mask.sum()) - 1:
            continue
        label_cells = np.argwhere(target_mask & (patch != major))
        if label_cells.shape[0] != 1:
            continue
        label_local = tuple(map(int, label_cells[0]))
        if label_local not in _opening_endpoints(patch.shape, direction):
            continue
        label_color = int(patch[label_local])
        matches.append(
            {
                "bbox": bbox_,
                "direction": direction,
                "border_color": major,
                "label_color": label_color,
                "label_pos": (r0 + label_local[0], c0 + label_local[1]),
            }
        )

    assert len(matches) <= 1, "container orientation should be unambiguous"
    return matches[0] if matches else None


def _flip_patch(patch: np.ndarray, direction: str) -> np.ndarray:
    return np.flipud(patch) if direction in ("up", "down") else np.fliplr(patch)


def _pattern_destination(
    container_bbox: tuple[int, int, int, int],
    container_direction: str,
    patch_shape: tuple[int, int],
) -> tuple[int, int]:
    r0, c0, r1, c1 = container_bbox
    ph, pw = patch_shape
    ir0, ic0, ih, iw = _interior_rect((r1 - r0 + 1, c1 - c0 + 1), container_direction)
    assert ph <= ih and pw <= iw, "pattern must fit inside its container"

    top = r0 + ir0
    left = c0 + ic0
    if container_direction == "up":
        return top + ih - ph, left + (iw - pw) // 2
    if container_direction == "down":
        return top, left + (iw - pw) // 2
    if container_direction == "left":
        return top + (ih - ph) // 2, left + iw - pw
    return top + (ih - ph) // 2, left


def _paste_patch(out: np.ndarray, patch: np.ndarray, row: int, col: int, bg: int) -> None:
    ph, pw = patch.shape
    region = out[row : row + ph, col : col + pw]
    mask = patch != bg
    assert not np.any((region != bg) & mask), "transformed patterns must not overlap"
    region[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pour each labeled outside pattern into the matching U-shaped container.

    Containers are matched by the unique label color on one opening endpoint.
    The opening direction chooses the mirror axis, and the flipped pattern is
    placed against the opposite inner wall, centered with upper/left bias.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)

    containers = []
    patterns = []
    for coords in _components(g != bg):
        container = _classify_container(g, coords, bg)
        if container is not None:
            containers.append(container)
            continue
        patch, bbox_ = _component_patch(g, coords, bg)
        patterns.append(
            {
                "bbox": bbox_,
                "coords": coords,
                "patch": patch,
                "colors": set(map(int, patch[patch != bg].ravel())),
            }
        )

    assert containers, "expected at least one container"
    assert len(containers) == len(patterns), "patterns and containers must pair one-to-one"

    pairs = []
    used_patterns: set[int] = set()
    for container in containers:
        matches = [idx for idx, pat in enumerate(patterns) if container["label_color"] in pat["colors"]]
        assert len(matches) == 1, "label colors must identify patterns uniquely"
        idx = matches[0]
        assert idx not in used_patterns, "pattern may match only one container"
        used_patterns.add(idx)
        pairs.append((container, patterns[idx]))

    out = g.copy()
    for pattern in patterns:
        out[pattern["coords"][:, 0], pattern["coords"][:, 1]] = bg
    for container in containers:
        lr, lc = container["label_pos"]
        out[lr, lc] = container["border_color"]
    for container, pattern in pairs:
        transformed = _flip_patch(pattern["patch"], container["direction"])
        row, col = _pattern_destination(container["bbox"], container["direction"], transformed.shape)
        _paste_patch(out, transformed, row, col, bg)
    return out


def _random_connected_mask(height: int, width: int, target_area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True
    frontier = set(_neighbors4(start[0], start[1], height, width))
    while int(mask.sum()) < target_area:
        if not frontier:
            return mask
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        mask[cell] = True
        for nb in _neighbors4(cell[0], cell[1], height, width):
            if not mask[nb]:
                frontier.add(nb)
    return mask


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _mask_is_valid_pattern(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if not (2 <= h <= 4 and 2 <= w <= 4):
        return False
    if mask.sum() < 3:
        return False
    if np.array_equal(mask, np.flipud(mask)) or np.array_equal(mask, np.fliplr(mask)):
        return False
    return not any(
        candidate is not None and np.array_equal(mask, candidate)
        for candidate in (_u_mask(mask.shape, direction) for direction in DIRECTIONS)
    )


def _random_pattern_patch(label_color: int, accent_choices: list[int]) -> np.ndarray | None:
    for _ in range(PATTERN_RETRIES):
        height = random.randint(2, 4)
        width = random.randint(2, 4)
        target_area = random.randint(3, height * width)
        mask = _tight_crop(_random_connected_mask(height, width, target_area))
        if not _mask_is_valid_pattern(mask):
            continue

        coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        usable_accents = [c for c in accent_choices if c != label_color]
        max_colors = min(3, len(usable_accents) + 1, len(coords))
        num_colors = 2 if max_colors == 2 else random.choice((2, 3))
        num_colors = min(num_colors, max_colors)
        chosen = [label_color] + random.sample(usable_accents, num_colors - 1)

        patch = np.full(mask.shape, BG, dtype=int)
        random.shuffle(coords)
        for color, cell in zip(chosen, coords, strict=False):
            patch[cell] = color
        for cell in coords[len(chosen) :]:
            patch[cell] = random.choice(chosen)

        if np.array_equal(patch, np.flipud(patch)) or np.array_equal(patch, np.fliplr(patch)):
            continue
        return patch
    return None


def _build_container_patch(
    direction: str,
    border_color: int,
    label_color: int,
    pattern_shape: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int]]:
    ph, pw = pattern_shape
    if direction in ("up", "down"):
        interior_h = ph + random.randint(1, 3)
        interior_w = pw + random.randint(0, 3)
        shape = (interior_h + 1, interior_w + 2)
    else:
        interior_h = ph + random.randint(0, 3)
        interior_w = pw + random.randint(1, 3)
        shape = (interior_h + 2, interior_w + 1)

    mask = _u_mask(shape, direction)
    assert mask is not None
    patch = np.full(shape, BG, dtype=int)
    patch[mask] = border_color
    label_local = random.choice(tuple(_opening_endpoints(shape, direction)))
    patch[label_local] = label_color
    return patch, label_local


def _all_positions(occupied: np.ndarray, height: int, width: int) -> list[tuple[int, int]]:
    h, w = occupied.shape
    out = []
    for r in range(h - height + 1):
        for c in range(w - width + 1):
            if not occupied[r : r + height, c : c + width].any():
                out.append((r, c))
    return out


def _reserve(occupied: np.ndarray, row: int, col: int, height: int, width: int) -> None:
    h, w = occupied.shape
    r0 = max(0, row - 1)
    c0 = max(0, col - 1)
    r1 = min(h, row + height + 1)
    c1 = min(w, col + width + 1)
    occupied[r0:r1, c0:c1] = True


def _expected_output(grid: np.ndarray, containers: list[dict], patterns: list[dict]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for pattern in patterns:
        pr, pc = pattern["pos"]
        patch = pattern["patch"]
        mask = patch != BG
        out[pr : pr + patch.shape[0], pc : pc + patch.shape[1]][mask] = BG
    for container in containers:
        lr, lc = container["label_pos"]
        out[lr, lc] = container["border_color"]
    for container, pattern in zip(containers, patterns, strict=False):
        transformed = _flip_patch(pattern["patch"], container["direction"])
        row, col = _pattern_destination(container["bbox"], container["direction"], transformed.shape)
        _paste_patch(out, transformed, row, col, BG)
    return out


def _generate_case() -> np.ndarray | None:
    grid_h = random.randint(20, 30)
    grid_w = random.randint(20, 30)
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    count = random.choice((2, 3, 4))
    directions = random.sample(DIRECTIONS, count)
    colors = list(COLOR_POOL)
    random.shuffle(colors)
    label_colors = colors[:count]
    border_colors = colors[count : 2 * count]
    accent_pool = [color for color in colors if color not in label_colors]

    containers: list[dict] = []
    patterns: list[dict] = []
    for idx, direction in enumerate(directions):
        pattern_patch = _random_pattern_patch(label_colors[idx], accent_pool)
        if pattern_patch is None:
            return None
        container_patch, label_local = _build_container_patch(
            direction, border_colors[idx], label_colors[idx], pattern_patch.shape
        )
        containers.append(
            {
                "direction": direction,
                "border_color": border_colors[idx],
                "label_color": label_colors[idx],
                "patch": container_patch,
                "label_local": label_local,
            }
        )
        patterns.append({"patch": pattern_patch, "label_color": label_colors[idx]})

    objects = []
    for idx, container in enumerate(containers):
        objects.append(
            {
                "kind": "container",
                "index": idx,
                "patch": container["patch"],
            }
        )
    for idx, pattern in enumerate(patterns):
        objects.append(
            {
                "kind": "pattern",
                "index": idx,
                "patch": pattern["patch"],
            }
        )
    random.shuffle(objects)
    objects.sort(key=lambda item: item["patch"].shape[0] * item["patch"].shape[1], reverse=True)

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    for obj in objects:
        ph, pw = obj["patch"].shape
        positions = _all_positions(occupied, ph, pw)
        if not positions:
            return None
        row, col = random.choice(positions)
        _reserve(occupied, row, col, ph, pw)
        obj["pos"] = (row, col)

    for obj in objects:
        row, col = obj["pos"]
        patch = obj["patch"]
        grid[row : row + patch.shape[0], col : col + patch.shape[1]] = patch
        target = containers if obj["kind"] == "container" else patterns
        target[obj["index"]]["pos"] = (row, col)

    for container in containers:
        row, col = container["pos"]
        ph, pw = container["patch"].shape
        container["bbox"] = (row, col, row + ph - 1, col + pw - 1)
        container["label_pos"] = (
            row + container["label_local"][0],
            col + container["label_local"][1],
        )

    try:
        expected = _expected_output(grid, containers, patterns)
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GENERATION_RETRIES} attempts")
