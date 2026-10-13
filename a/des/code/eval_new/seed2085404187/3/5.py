from __future__ import annotations

import random

import numpy as np


BLACK = 0
MAROON = 9
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8]
SIDES = ("top", "right", "bottom", "left")
ROTATION_K = {"top": 0, "right": 3, "bottom": 2, "left": 1}
GENERATE_RETRIES = 300
PLACEMENT_RETRIES = 400
TEMPLATE_RETRIES = 200


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _gap_index(n: int) -> int:
    return (int(n) - 1) // 2


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _classify_component(local_mask: np.ndarray) -> tuple[str | None, str | None]:
    h, w = map(int, local_mask.shape)

    if h == w and h in (3, 4) and bool(local_mask.all()):
        return "target", None

    if h != w or h not in (5, 6):
        return None, None

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if bool(np.any(local_mask & ~border)):
        return None, None

    missing = np.argwhere(border & ~local_mask)
    if int(len(missing)) != 1:
        return None, None

    gap = tuple(map(int, missing[0]))
    side_to_gap = {
        "top": (0, _gap_index(w)),
        "right": (_gap_index(h), w - 1),
        "bottom": (h - 1, _gap_index(w)),
        "left": (_gap_index(h), 0),
    }
    for side, pos in side_to_gap.items():
        if gap == pos and int(local_mask.sum()) == int(border.sum()) - 1:
            return "ring", side
    return None, None


def _rotate_template(mask: np.ndarray, side: str) -> np.ndarray:
    return np.rot90(mask, ROTATION_K[side])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract maroon templates from gapped color rings and print the rotated result onto matching targets.

    Input structure:
    - Black background with several colored 1-pixel-thick square rings whose inner size is 3x3 or 4x4.
    - Each ring border has exactly one missing pixel on the midpoint of one side.
    - Inside every ring sits a maroon template shape on black cells.
    - Elsewhere, each ring color also appears as a separate solid square target block matching that ring's inner size.

    Rule:
    - Identify each ring and read the maroon template inside it.
    - Match the template to the solid target block that has the same color and size as the ring.
    - Decode the ring gap as a rotation command: top = 0 degrees, right = 90 degrees clockwise, bottom = 180 degrees, left = 270 degrees clockwise.
    - Remove the source rings and maroon templates, and paint the rotated template onto the matched target using the ring color on a black background.

    Key insight:
    - The colored ring supplies both the destination key and the rotation command: its color selects the target block, while its gap side selects how the hidden maroon template must be rotated.
    """
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)

    rings: dict[tuple[int, int, int], list[dict[str, object]]] = {}
    targets: dict[tuple[int, int, int], list[dict[str, object]]] = {}

    colors = sorted(int(x) for x in np.unique(g) if int(x) not in (BLACK, MAROON))
    for color in colors:
        for coords in _components(g == int(color)):
            r0, c0, r1, c1 = _bbox(coords)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            local = np.zeros((h, w), dtype=bool)
            local[coords[:, 0] - r0, coords[:, 1] - c0] = True
            kind, side = _classify_component(local)
            if kind == "ring":
                inner = g[r0 + 1 : r1, c0 + 1 : c1] == MAROON
                key = (int(color), int(inner.shape[0]), int(inner.shape[1]))
                rings.setdefault(key, []).append(
                    {
                        "bbox": (int(r0), int(c0), int(r1), int(c1)),
                        "side": str(side),
                        "template": inner.copy(),
                    }
                )
            elif kind == "target":
                key = (int(color), int(h), int(w))
                targets.setdefault(key, []).append({"bbox": (int(r0), int(c0), int(r1), int(c1))})

    assert rings, "at least one ring is required"
    assert set(rings) == set(targets), "ring and target keys must match"

    for key, ring_list in rings.items():
        target_list = targets[key]
        assert len(ring_list) == len(target_list), "ring/target counts must match per color+size"
        ring_list.sort(key=lambda item: tuple(item["bbox"]))  # deterministic if duplicates exist
        target_list.sort(key=lambda item: tuple(item["bbox"]))
        color = int(key[0])
        for ring, target in zip(ring_list, target_list):
            template = np.asarray(ring["template"], dtype=bool)
            rotated = _rotate_template(template, str(ring["side"]))
            r0, c0, r1, c1 = map(int, target["bbox"])
            patch = out[r0 : r1 + 1, c0 : c1 + 1]
            assert patch.shape == rotated.shape
            patch[rotated] = color

    return out


def _random_template(size: int) -> np.ndarray:
    min_cells = 3
    max_cells = min(size * size - 2, size + 2)
    for _ in range(TEMPLATE_RETRIES):
        goal = random.randint(min_cells, max_cells)
        start = (random.randrange(size), random.randrange(size))
        cells = {start}
        frontier = set()
        for nr, nc in _neighbors4(*start):
            if 0 <= nr < size and 0 <= nc < size:
                frontier.add((int(nr), int(nc)))
        while len(cells) < goal and frontier:
            chosen = random.choice(tuple(frontier))
            frontier.remove(chosen)
            cells.add(chosen)
            for nr, nc in _neighbors4(*chosen):
                if 0 <= nr < size and 0 <= nc < size and (int(nr), int(nc)) not in cells:
                    frontier.add((int(nr), int(nc)))
        if len(cells) != goal:
            continue
        mask = np.zeros((size, size), dtype=bool)
        for r, c in cells:
            mask[int(r), int(c)] = True
        if int(np.any(mask, axis=0).sum()) < 2 or int(np.any(mask, axis=1).sum()) < 2:
            continue
        if len({np.rot90(mask, k).tobytes() for k in range(4)}) != 4:
            continue
        return mask
    raise RuntimeError("failed to generate a non-symmetric template")


def _ring_patch(color: int, size: int, side: str) -> np.ndarray:
    outer = size + 2
    patch = np.zeros((outer, outer), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    gap_positions = {
        "top": (0, _gap_index(outer)),
        "right": (_gap_index(outer), outer - 1),
        "bottom": (outer - 1, _gap_index(outer)),
        "left": (_gap_index(outer), 0),
    }
    gr, gc = gap_positions[side]
    patch[int(gr), int(gc)] = BLACK
    return patch


def _can_place(used: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    H, W = map(int, used.shape)
    r1 = int(r0 + h - 1)
    c1 = int(c0 + w - 1)
    rr0 = max(0, int(r0 - pad))
    cc0 = max(0, int(c0 - pad))
    rr1 = min(H - 1, int(r1 + pad))
    cc1 = min(W - 1, int(c1 + pad))
    return not bool(used[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _reserve(used: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    H, W = map(int, used.shape)
    r1 = int(r0 + h - 1)
    c1 = int(c0 + w - 1)
    rr0 = max(0, int(r0 - pad))
    cc0 = max(0, int(c0 - pad))
    rr1 = min(H - 1, int(r1 + pad))
    cc1 = min(W - 1, int(c1 + pad))
    used[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _sample_sizes(n_pairs: int) -> list[int]:
    sizes = [3, 4]
    while len(sizes) < n_pairs:
        sizes.append(random.choice([3, 4]))
    random.shuffle(sizes)
    return sizes[:n_pairs]


def _gen_attempt() -> np.ndarray | None:
    n_pairs = random.choice([2, 3, 4])
    sizes = _sample_sizes(n_pairs)
    colors = random.sample(COLOR_POOL, n_pairs)
    sides = random.sample(list(SIDES), n_pairs)

    specs: list[dict[str, object]] = []
    for color, size, side in zip(colors, sizes, sides):
        specs.append(
            {
                "color": int(color),
                "size": int(size),
                "side": str(side),
                "template": _random_template(int(size)),
            }
        )

    min_dim = min(30, 12 + 4 * n_pairs)
    H = random.randint(min_dim, 30)
    W = random.randint(min_dim, 30)
    used = np.zeros((H, W), dtype=bool)

    items: list[tuple[str, int, int, int]] = []
    for idx, spec in enumerate(specs):
        size = int(spec["size"])
        items.append(("ring", int(idx), size + 2, size + 2))
        items.append(("target", int(idx), size, size))
    random.shuffle(items)

    placements: dict[tuple[int, str], tuple[int, int]] = {}
    for kind, idx, h, w in items:
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = random.randint(0, H - h)
            c0 = random.randint(0, W - w)
            if not _can_place(used, int(r0), int(c0), int(h), int(w), pad=1):
                continue
            _reserve(used, int(r0), int(c0), int(h), int(w), pad=1)
            placements[(int(idx), str(kind))] = (int(r0), int(c0))
            placed = True
            break
        if not placed:
            return None

    grid = np.zeros((H, W), dtype=int)
    expected = np.zeros_like(grid)

    for idx, spec in enumerate(specs):
        color = int(spec["color"])
        size = int(spec["size"])
        side = str(spec["side"])
        template = np.asarray(spec["template"], dtype=bool)

        rr, rc = placements[(int(idx), "ring")]
        ring = _ring_patch(color, size, side)
        ring_view = grid[rr : rr + size + 2, rc : rc + size + 2]
        ring_view[ring != BLACK] = ring[ring != BLACK]
        inner = grid[rr + 1 : rr + 1 + size, rc + 1 : rc + 1 + size]
        inner[template] = MAROON

        tr, tc = placements[(int(idx), "target")]
        grid[tr : tr + size, tc : tc + size] = color

        rotated = _rotate_template(template, side)
        expected[tr : tr + size, tc : tc + size][rotated] = color

    try:
        result = main(grid)
    except Exception:
        return None
    return grid if np.array_equal(result, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATE_RETRIES} attempts")
