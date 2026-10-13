import random
from collections import Counter

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = list(range(1, 10))


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
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
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_border(coords: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _tight_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(map(tuple, np.argwhere(mask))))


def _hole_count(mask: np.ndarray) -> int:
    holes = 0
    for comp in _components(~mask):
        if not _touches_border(comp, mask.shape):
            holes += 1
    return holes


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps = []
    for color in sorted(int(x) for x in np.unique(grid) if int(x) != bg):
        for coords in _components(grid == color):
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": len(coords),
                }
            )
    return comps


def _extract_containers(grid: np.ndarray, bg: int) -> list[dict]:
    containers = []
    for comp in _monochrome_components(grid, bg):
        r0, c0, r1, c1 = comp["bbox"]
        if (r1 - r0 + 1) < 4 or (c1 - c0 + 1) < 4:
            continue
        if comp["area"] < 8:
            continue
        color = comp["color"]
        crop = grid[r0 : r1 + 1, c0 : c1 + 1]

        cavity_local: list[tuple[int, int]] = []
        for bg_comp in _components(crop == bg):
            if not _touches_border(bg_comp, crop.shape):
                cavity_local.extend(bg_comp)
        if not cavity_local:
            continue

        pillar_count = 0
        protected = []
        for color_comp in _components(crop == color):
            touches = _touches_border(color_comp, crop.shape)
            if not touches:
                pillar_count += 1
            for rr, cc in color_comp:
                protected.append((r0 + rr, c0 + cc))

        cavity_mask, cavity_bbox_local = _tight_mask(cavity_local)
        cr0, cc0, cr1, cc1 = cavity_bbox_local
        containers.append(
            {
                "color": color,
                "topology": pillar_count,
                "shape_key": _shape_key(cavity_mask),
                "cavity_bbox": (r0 + cr0, c0 + cc0, r0 + cr1, c0 + cc1),
                "protected": protected,
            }
        )
    return containers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each scattered sprite into the monochrome cavity whose enclosed pillar count matches the sprite's hole count.

    Input structure:
    - Black background surrounds 2-3 large monochrome shell objects and 2-3 smaller scattered multicolor sprites.
    - Each large shell encloses one background cavity; some cavities also contain enclosed same-color pillar islands.
    - Each small sprite is a single 4-connected object outside the large shells and may itself contain enclosed background holes.

    Rule:
    - Detect every large shell's enclosed cavity and count how many enclosed same-color pillar islands lie inside it.
    - Detect every scattered sprite and count its enclosed holes inside its tight bounding box.
    - Match sprites to cavities by equal hole-count / pillar-count, then translate each sprite without rotation so its foreground exactly covers that cavity's background cells.
    - Clear the sprite's original location and keep the shell objects plus their enclosed pillars unchanged.

    Key insight:
    - The destination is chosen by a topological signature rather than by color or position: sprite hole count selects the only cavity with the same enclosed-pillar topology, and geometry fixes the translation.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    containers = _extract_containers(g, bg)
    assert containers, "expected at least one valid container"

    protected = np.zeros(g.shape, dtype=bool)
    for container in containers:
        for r, c in container["protected"]:
            protected[r, c] = True

    sprite_mask = (g != bg) & ~protected
    sprite_comps = _components(sprite_mask)
    assert sprite_comps, "expected scattered sprites"

    sprites = []
    for coords in sprite_comps:
        mask, bbox = _tight_mask(coords)
        sprites.append(
            {
                "coords": coords,
                "bbox": bbox,
                "hole_count": _hole_count(mask),
                "shape_key": _shape_key(mask),
            }
        )

    assert len(sprites) == len(containers), "sprite/container counts must match"
    out = g.copy()
    out[sprite_mask] = bg
    used = set()

    for sprite in sprites:
        candidates = [
            idx
            for idx, container in enumerate(containers)
            if idx not in used
            and container["topology"] == sprite["hole_count"]
            and container["shape_key"] == sprite["shape_key"]
        ]
        assert len(candidates) == 1, "topology + shape should identify exactly one cavity"
        idx = candidates[0]
        used.add(idx)
        sr0, sc0, _sr1, _sc1 = sprite["bbox"]
        cr0, cc0, _cr1, _cc1 = containers[idx]["cavity_bbox"]
        dr = cr0 - sr0
        dc = cc0 - sc0
        for r, c in sprite["coords"]:
            out[r + dr, c + dc] = g[r, c]

    return out


def _mask_from_rows(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


SHAPE_LIBRARY = {
    0: [
        _mask_from_rows(["111", "110"]),
        _mask_from_rows(["110", "011"]),
        _mask_from_rows(["111", "010"]),
        _mask_from_rows(["111", "001"]),
        _mask_from_rows(["010", "111", "010"]),
    ],
    1: [
        _mask_from_rows(["111", "101", "111"]),
        _mask_from_rows(["1111", "1001", "1111"]),
        _mask_from_rows(["111", "101", "111", "001"]),
        _mask_from_rows(["111", "101", "111", "100"]),
    ],
    2: [
        _mask_from_rows(["11111", "10101", "11111"]),
        _mask_from_rows(["111", "101", "111", "101", "111"]),
    ],
}


def _randomize_mask(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return out


def _colorize_sprite(mask: np.ndarray, colors: list[int]) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    coords = list(map(tuple, np.argwhere(mask)))
    palette = random.sample(colors, k=random.randint(2, min(4, len(colors))))
    style = random.choice(["checker", "rows", "cols", "random"])
    for r, c in coords:
        if style == "checker":
            color = palette[(r + c) % len(palette)]
        elif style == "rows":
            color = palette[r % len(palette)]
        elif style == "cols":
            color = palette[c % len(palette)]
        else:
            color = random.choice(palette)
        patch[r, c] = color
    return patch


def _build_container(sprite_mask: np.ndarray, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    sh, sw = sprite_mask.shape
    top = random.randint(1, 2)
    bottom = random.randint(1, 2)
    left = random.randint(1, 2)
    right = random.randint(1, 2)
    patch = np.full((top + sh + bottom, left + sw + right), color, dtype=int)
    for r, c in np.argwhere(sprite_mask):
        patch[top + r, left + c] = 0
    return patch, (top, left)


def _make_valid_pair(hole_count: int, shell_color: int, sprite_colors: list[int]) -> dict:
    for _ in range(200):
        sprite_mask = _randomize_mask(random.choice(SHAPE_LIBRARY[hole_count]))
        if len(_components(sprite_mask)) != 1:
            continue
        if _hole_count(sprite_mask) != hole_count:
            continue
        sprite_patch = _colorize_sprite(sprite_mask, sprite_colors)
        used_colors = {int(x) for x in np.unique(sprite_patch) if int(x) != 0}
        if len(used_colors) < 2:
            continue
        container_patch, cavity_offset = _build_container(sprite_mask, shell_color)
        containers = _extract_containers(container_patch, 0)
        if len(containers) != 1:
            continue
        container = containers[0]
        if container["topology"] != hole_count:
            continue
        if container["shape_key"] != _shape_key(sprite_mask):
            continue
        return {
            "holes": hole_count,
            "sprite_patch": sprite_patch,
            "container_patch": container_patch,
            "cavity_offset": cavity_offset,
        }
    raise RuntimeError("failed to build a valid sprite/container pair")


def _can_place(blocked: np.ndarray, h: int, w: int, r0: int, c0: int) -> bool:
    hh, ww = blocked.shape
    if r0 < 0 or c0 < 0 or r0 + h > hh or c0 + w > ww:
        return False
    return not blocked[r0 : r0 + h, c0 : c0 + w].any()


def _mark_blocked(blocked: np.ndarray, h: int, w: int, r0: int, c0: int) -> None:
    hh, ww = blocked.shape
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(hh, r0 + h + 1)
    cc1 = min(ww, c0 + w + 1)
    blocked[rr0:rr1, cc0:cc1] = True


def _place_patch(
    grid: np.ndarray, blocked: np.ndarray, patch: np.ndarray
) -> tuple[int, int] | None:
    h, w = patch.shape
    hh, ww = grid.shape
    cells = [(r, c) for r in range(hh - h + 1) for c in range(ww - w + 1)]
    random.shuffle(cells)
    for r0, c0 in cells:
        if not _can_place(blocked, h, w, r0, c0):
            continue
        mask = patch != 0
        view = grid[r0 : r0 + h, c0 : c0 + w]
        view[mask] = patch[mask]
        _mark_blocked(blocked, h, w, r0, c0)
        return r0, c0
    return None


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    pair_count = 3 if random.random() < 0.6 else 2
    counts = random.sample([0, 1, 2], k=pair_count)
    shell_colors = random.sample(PALETTE, k=pair_count)
    sprite_colors = [c for c in PALETTE if c not in shell_colors]

    pairs = []
    widths = 0
    heights = 0
    for hole_count, shell_color in zip(counts, shell_colors):
        pair = _make_valid_pair(hole_count, shell_color, sprite_colors)
        pairs.append(pair)
        container_patch = pair["container_patch"]
        sprite_patch = pair["sprite_patch"]
        widths += container_patch.shape[1] + sprite_patch.shape[1] + 3
        heights = max(heights, container_patch.shape[0] + sprite_patch.shape[0] + 4)

    min_h = max(20, heights + 4)
    min_w = max(20, min(28, widths // 2 + 5))
    h = random.randint(min_h, min(30, max(min_h, 26)))
    w = random.randint(min_w, min(30, max(min_w, 28)))
    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    container_positions = []
    sprite_positions = []
    items = sorted(enumerate(pairs), key=lambda x: x[1]["container_patch"].shape[0] * x[1]["container_patch"].shape[1], reverse=True)

    for idx, pair in items:
        pos = _place_patch(grid, blocked, pair["container_patch"])
        if pos is None:
            raise RuntimeError("failed to place container")
        container_positions.append((idx, pos))

    random.shuffle(items)
    for idx, pair in items:
        pos = _place_patch(grid, blocked, pair["sprite_patch"])
        if pos is None:
            raise RuntimeError("failed to place sprite")
        sprite_positions.append((idx, pos))

    container_positions = dict(container_positions)
    sprite_positions = dict(sprite_positions)

    out = grid.copy()
    for idx, pair in enumerate(pairs):
        sr0, sc0 = sprite_positions[idx]
        sprite = pair["sprite_patch"]
        smask = sprite != 0
        out[sr0 : sr0 + sprite.shape[0], sc0 : sc0 + sprite.shape[1]][smask] = 0

        cr0, cc0 = container_positions[idx]
        off_r, off_c = pair["cavity_offset"]
        dest = out[cr0 + off_r : cr0 + off_r + sprite.shape[0], cc0 + off_c : cc0 + off_c + sprite.shape[1]]
        dest[smask] = sprite[smask]

    return grid, out


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            inp, expected = _generate_case()
        except RuntimeError:
            continue
        produced = main(inp)
        if np.array_equal(produced, expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
