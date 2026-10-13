from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

GEN_ATTEMPTS = 500
MASK_ATTEMPTS = 200
LAYOUT_ATTEMPTS = 300
ROW_GAP = 1

STRUCTURE_CASES = [
    ((2, 2), 1),
    ((2, 2), 2),
    ((2, 2), 3),
    ((2, 3), 0),
    ((2, 3), 1),
    ((2, 3), 2),
    ((3, 3), 0),
    ((3, 3), 1),
    ((2, 2, 2), 0),
    ((2, 2, 2), 1),
    ((2, 2, 3), 0),
]
STRUCTURE_WEIGHTS = [2, 3, 1, 2, 3, 1, 2, 1, 1, 1, 1]

WIDTH_CHOICES = [1, 2, 2, 3, 3, 4]
HEIGHT_CHOICES = [1, 1, 2, 2, 3]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    colors = set(map(int, np.unique(g)))
    assert colors <= {BG, RED}, "input must contain only black background and red objects"
    comps = _components(g == RED)
    assert comps, "expected at least one red object"

    objects = []
    for coords in comps:
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        objects.append(
            {
                "coords": coords,
                "bbox": box,
                "left": c0,
                "width": c1 - c0 + 1,
            }
        )
    return objects


def _link_graph(objects: list[dict]) -> tuple[list[list[int]], list[list[int]]]:
    by_left: defaultdict[int, list[int]] = defaultdict(list)
    for idx, obj in enumerate(objects):
        by_left[int(obj["left"])].append(idx)

    preds = [[] for _ in objects]
    succs = [[] for _ in objects]
    for idx, obj in enumerate(objects):
        target = int(obj["left"]) + int(obj["width"])
        for nxt in by_left.get(target, []):
            succs[idx].append(nxt)
            preds[nxt].append(idx)

    for idx in range(len(objects)):
        assert len(preds[idx]) <= 1 and len(succs[idx]) <= 1, "chain layout must be unambiguous"
    return preds, succs


def _paint_object(
    out: np.ndarray,
    occupied: np.ndarray,
    mask: np.ndarray,
    top: int,
    left: int,
    color: int,
) -> None:
    h, w = mask.shape
    assert 0 <= top and 0 <= left and top + h <= out.shape[0] and left + w <= out.shape[1]
    view = occupied[top : top + h, left : left + w]
    assert not bool(view[mask].any()), "objects must not overlap"
    out[top : top + h, left : left + w][mask] = color
    view[mask] = True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift horizontally adjacent bbox-linked objects right by their own widths and recolor them by chain role.

    Input structure:
    - The grid has a black background and several small red 8-connected foreground objects.
    - Each object's bounding box has a well-defined left edge and width.
    - Some objects form horizontal bbox-adjacency chains where one bbox starts exactly where the previous bbox ends.

    Rule:
    - Find every red 8-connected object and compute its bounding-box width W and left column L.
    - Add a directed link A->B when B.left == A.left + A.width, then extract every maximal chain.
    - Redraw from a blank black canvas: isolated objects stay red in place, while every chain object shifts right by its own width and is recolored blue/green/yellow for head/middle/tail.

    Key insight:
    - The same bbox width controls both the chain connection test and the object's personal translation distance, while the object's position inside a maximal chain determines its output color.
    """
    g = np.asarray(grid, dtype=int)
    objects = _extract_objects(g)
    preds, succs = _link_graph(objects)

    chain_ids: set[int] = set()
    chains: list[list[int]] = []
    for idx in range(len(objects)):
        if preds[idx] or not succs[idx]:
            continue
        chain = [idx]
        cur = idx
        while succs[cur]:
            nxt = succs[cur][0]
            chain.append(nxt)
            cur = nxt
        assert len(chain) >= 2
        chains.append(chain)
        chain_ids.update(chain)

    for idx in range(len(objects)):
        if preds[idx] or succs[idx]:
            assert idx in chain_ids, "linked objects must belong to some maximal chain"

    isolated = [idx for idx in range(len(objects)) if idx not in chain_ids]
    isolated.sort(key=lambda idx: (objects[idx]["bbox"][0], objects[idx]["bbox"][1]))
    chains.sort(key=lambda chain: (objects[chain[0]]["bbox"][0], objects[chain[0]]["bbox"][1]))

    out = np.full_like(g, BG)
    occupied = np.zeros(g.shape, dtype=bool)

    for idx in isolated:
        obj = objects[idx]
        r0, c0, r1, c1 = obj["bbox"]
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[obj["coords"][:, 0] - r0, obj["coords"][:, 1] - c0] = True
        _paint_object(out, occupied, mask, r0, c0, RED)

    for chain in chains:
        for pos, idx in enumerate(chain):
            obj = objects[idx]
            r0, c0, r1, c1 = obj["bbox"]
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[obj["coords"][:, 0] - r0, obj["coords"][:, 1] - c0] = True
            if pos == 0:
                color = BLUE
            elif pos == len(chain) - 1:
                color = YELLOW
            else:
                color = GREEN
            _paint_object(out, occupied, mask, r0, c0 + int(obj["width"]), color)

    return out


def _random_partition(total: int, parts: int) -> list[int]:
    values = [0] * parts
    for _ in range(total):
        values[random.randrange(parts)] += 1
    return values


def _random_mask(height: int, width: int) -> np.ndarray | None:
    min_area = max(2, max(height, width))
    max_area = min(6, height * width)
    if min_area > max_area:
        return None

    cells = [(r, c) for r in range(height) for c in range(width)]
    for _ in range(MASK_ATTEMPTS):
        area = random.randint(min_area, max_area)
        chosen = {random.choice(cells)}
        while len(chosen) < area:
            frontier = [
                cell
                for cell in cells
                if cell not in chosen
                and any(max(abs(cell[0] - r), abs(cell[1] - c)) == 1 for r, c in chosen)
            ]
            if not frontier:
                break
            chosen.add(random.choice(frontier))

        if len(chosen) != area:
            continue

        rows = [r for r, _ in chosen]
        cols = [c for _, c in chosen]
        if min(rows) != 0 or max(rows) != height - 1:
            continue
        if min(cols) != 0 or max(cols) != width - 1:
            continue

        mask = np.zeros((height, width), dtype=bool)
        for r, c in chosen:
            mask[r, c] = True
        return mask
    return None


def _make_object() -> dict | None:
    for _ in range(MASK_ATTEMPTS):
        width = random.choice(WIDTH_CHOICES)
        height = random.choice(HEIGHT_CHOICES)
        mask = _random_mask(height, width)
        if mask is None:
            continue
        return {
            "mask": mask,
            "width": width,
            "height": height,
            "area": int(mask.sum()),
        }
    return None


def _assign_rows(specs: list[dict]) -> int | None:
    order = list(range(len(specs)))
    random.shuffle(order)
    min_height = sum(int(spec["height"]) for spec in specs) + ROW_GAP * (len(specs) - 1)
    if min_height > 30:
        return None

    height = random.randint(max(9, min_height), min(30, min_height + 6))
    slack = _random_partition(height - min_height, len(specs) + 1)

    row = slack[0]
    for slot, idx in enumerate(order):
        specs[idx]["top"] = row
        row += int(specs[idx]["height"])
        if slot < len(order) - 1:
            row += ROW_GAP + slack[slot + 1]
    return height


def _compute_chain_need(chain: list[int], specs: list[dict]) -> int:
    prefix = 0
    need = 0
    for idx in chain:
        width = int(specs[idx]["width"])
        need = max(need, prefix + 2 * width)
        prefix += width
    return need


def _valid_links(specs: list[dict], chains: list[list[int]]) -> bool:
    intended: dict[int, int] = {}
    for chain in chains:
        for a, b in zip(chain, chain[1:]):
            intended[a] = b

    for idx, spec in enumerate(specs):
        target = int(spec["left"]) + int(spec["width"])
        matches = [j for j, other in enumerate(specs) if int(other["left"]) == target]
        expected = [intended[idx]] if idx in intended else []
        if matches != expected:
            return False
    return True


def _assign_columns(specs: list[dict], chains: list[list[int]], width: int) -> bool:
    chain_members = {idx for chain in chains for idx in chain}
    isolated = [idx for idx in range(len(specs)) if idx not in chain_members]

    for _ in range(LAYOUT_ATTEMPTS):
        used_lefts: set[int] = set()
        for spec in specs:
            spec.pop("left", None)

        ok = True
        chain_order = list(range(len(chains)))
        random.shuffle(chain_order)
        for chain_idx in chain_order:
            chain = chains[chain_idx]
            max_start = width - _compute_chain_need(chain, specs)
            if max_start < 0:
                ok = False
                break

            starts = list(range(max_start + 1))
            random.shuffle(starts)
            placed = False
            for start in starts:
                prefix = 0
                trial_lefts = []
                for idx in chain:
                    left = start + prefix
                    prefix += int(specs[idx]["width"])
                    if left in used_lefts:
                        break
                    trial_lefts.append(left)
                else:
                    for idx, left in zip(chain, trial_lefts):
                        specs[idx]["left"] = left
                        used_lefts.add(left)
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        random.shuffle(isolated)
        for idx in isolated:
            max_left = width - int(specs[idx]["width"])
            candidates = [left for left in range(max_left + 1) if left not in used_lefts]
            random.shuffle(candidates)
            if not candidates:
                ok = False
                break
            specs[idx]["left"] = candidates[0]
            used_lefts.add(candidates[0])
        if not ok:
            continue

        if _valid_links(specs, chains):
            return True
    return False


def _build_grid(height: int, width: int, specs: list[dict]) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for spec in specs:
        top = int(spec["top"])
        left = int(spec["left"])
        mask = spec["mask"]
        grid[top : top + mask.shape[0], left : left + mask.shape[1]][mask] = RED
    return grid


def _expected_output(height: int, width: int, specs: list[dict], chains: list[list[int]]) -> np.ndarray:
    out = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    chain_members = {idx for chain in chains for idx in chain}

    isolated = [idx for idx in range(len(specs)) if idx not in chain_members]
    isolated.sort(key=lambda idx: (int(specs[idx]["top"]), int(specs[idx]["left"])))
    sorted_chains = sorted(chains, key=lambda chain: (int(specs[chain[0]]["top"]), int(specs[chain[0]]["left"])))

    for idx in isolated:
        spec = specs[idx]
        _paint_object(out, occupied, spec["mask"], int(spec["top"]), int(spec["left"]), RED)

    for chain in sorted_chains:
        for pos, idx in enumerate(chain):
            spec = specs[idx]
            if pos == 0:
                color = BLUE
            elif pos == len(chain) - 1:
                color = YELLOW
            else:
                color = GREEN
            _paint_object(
                out,
                occupied,
                spec["mask"],
                int(spec["top"]),
                int(spec["left"]) + int(spec["width"]),
                color,
            )
    return out


def _gen_attempt() -> np.ndarray | None:
    chain_lengths, isolated_count = random.choices(STRUCTURE_CASES, weights=STRUCTURE_WEIGHTS, k=1)[0]

    specs: list[dict] = []
    chains: list[list[int]] = []
    for chain_len in chain_lengths:
        chain = []
        for _ in range(chain_len):
            spec = _make_object()
            if spec is None:
                return None
            chain.append(len(specs))
            specs.append(spec)
        chains.append(chain)

    for _ in range(isolated_count):
        spec = _make_object()
        if spec is None:
            return None
        specs.append(spec)

    height = _assign_rows(specs)
    if height is None:
        return None

    min_width = max(18, max(_compute_chain_need(chain, specs) for chain in chains))
    if min_width > 30:
        return None
    width = random.randint(min_width, min(30, min_width + 8))
    if not _assign_columns(specs, chains, width):
        return None

    grid = _build_grid(height, width, specs)
    objects = _extract_objects(grid)
    if len(objects) != len(specs):
        return None

    expected = _expected_output(height, width, specs, chains)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
