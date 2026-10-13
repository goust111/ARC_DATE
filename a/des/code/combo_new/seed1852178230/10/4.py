from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 300
SCRATCH_SIZE = 30


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return mask[:0, :0].copy()
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _component_masks(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            r0, c0, r1, c1 = _bbox_from_coords(coords)
            comp = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                comp[rr - r0, cc - c0] = True
            comps.append(comp)
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _extract_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if g[r, c] == BG or seen[r, c]:
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if g[nr, nc] == color and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            r0, c0, r1, c1 = _bbox_from_coords(coords)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            comps.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "area": int(mask.sum()),
                }
            )
    return comps


def _downsample_uniform_blocks(mask: np.ndarray, scale: int) -> np.ndarray | None:
    h, w = mask.shape
    if h % scale != 0 or w % scale != 0:
        return None
    out = np.zeros((h // scale, w // scale), dtype=bool)
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            block = mask[r * scale : (r + 1) * scale, c * scale : (c + 1) * scale]
            if block.all():
                out[r, c] = True
            elif block.any():
                return None
    return out


def _candidate_pairs(comps: list[dict]) -> list[dict]:
    cands: list[dict] = []
    for big_idx, big in enumerate(comps):
        for scale in (2, 3):
            full_small = _downsample_uniform_blocks(big["mask"], scale)
            if full_small is None:
                continue
            for small_idx, small in enumerate(comps):
                if small_idx == big_idx:
                    continue
                if small["mask"].shape != full_small.shape:
                    continue
                if int(small["area"]) >= int(full_small.sum()):
                    continue
                if np.any(small["mask"] & ~full_small):
                    continue
                missing = full_small & ~small["mask"]
                if not missing.any():
                    continue
                cands.append(
                    {
                        "small": small_idx,
                        "big": big_idx,
                        "scale": scale,
                        "missing": missing,
                        "full_small": full_small,
                    }
                )
    return cands


def _unique_matching(comps: list[dict], cands: list[dict]) -> list[dict]:
    by_node: dict[int, list[int]] = {i: [] for i in range(len(comps))}
    for idx, cand in enumerate(cands):
        by_node[cand["small"]].append(idx)
        by_node[cand["big"]].append(idx)

    solutions: list[list[dict]] = []

    def rec(unused: frozenset[int], chosen: list[dict]) -> None:
        if len(solutions) >= 2:
            return
        if not unused:
            solutions.append(chosen.copy())
            return

        node = min(unused)
        options = [
            cands[idx]
            for idx in by_node[node]
            if cands[idx]["small"] in unused and cands[idx]["big"] in unused
        ]
        options.sort(
            key=lambda cand: (
                0 if cand["small"] == node else 1,
                -int(cand["full_small"].sum()),
                -int(cand["missing"].sum()),
                -int(cand["scale"]),
            )
        )
        for cand in options:
            nxt = unused - {cand["small"], cand["big"]}
            rec(nxt, chosen + [cand])

    rec(frozenset(range(len(comps))), [])
    assert len(solutions) == 1, "pairing must be unique"
    return solutions[0]


def _apply_pairs(grid: np.ndarray, comps: list[dict], pairs: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for cand in pairs:
        small = comps[cand["small"]]
        big = comps[cand["big"]]
        sr0, sc0, sr1, sc1 = small["bbox"]
        br0, bc0, _br1, _bc1 = big["bbox"]

        small_patch = out[sr0 : sr1 + 1, sc0 : sc1 + 1]
        small_patch[small["mask"]] = int(big["color"])

        scale = int(cand["scale"])
        for rr, cc in np.argwhere(cand["missing"]):
            r = int(rr)
            c = int(cc)
            out[
                br0 + r * scale : br0 + (r + 1) * scale,
                bc0 + c * scale : bc0 + (c + 1) * scale,
            ] = BG
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each holed small shape with its matching large shape's color and carve the large shape by the small hole pattern.

    Input structure:
    - Background is black.
    - Foreground consists of an even number of disconnected monochrome objects with distinct colors.
    - The objects form unique pairs: in each pair one object is a complete shape scaled by 2 or 3, while the smaller object fits inside the same unscaled shape but is missing some interior pixels.

    Rule:
    - Find the unique pairing by checking which larger object becomes a clean unscaled shape after blockwise downsampling by 2 or 3.
    - Match it with the smaller object whose pixels are a strict subset of that unscaled shape.
    - Recolor every remaining pixel of the smaller object to the larger object's color.
    - For every pixel missing from the smaller object, erase the corresponding scale-by-scale block inside the larger object to black.

    Key insight:
    - Pairing depends on scale-normalized shape agreement, while the transformation swaps two different attributes: color flows from large to small, and hole structure flows from small to large after scaling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    comps = _extract_components(g)
    assert len(comps) >= 2 and len(comps) % 2 == 0

    cands = _candidate_pairs(comps)
    pairs = _unique_matching(comps, cands)
    return _apply_pairs(g, comps, pairs)


def _interior_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not mask[r, c]:
                continue
            if mask[r - 1, c] and mask[r + 1, c] and mask[r, c - 1] and mask[r, c + 1]:
                cells.append((r, c))
    return cells


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    trimmed = _trim_mask(mask)
    return tuple(map(int, trimmed.shape)), trimmed.tobytes()


def _try_boundary_cut(mask: np.ndarray, rng: random.Random) -> np.ndarray | None:
    h, w = mask.shape
    kind = rng.choice(("corner", "edge"))
    nxt = mask.copy()

    if kind == "corner":
        corner = rng.choice(("tl", "tr", "bl", "br"))
        dh = rng.randint(1, min(2, h - 1))
        dw = rng.randint(1, min(2, w - 1))
        if corner == "tl":
            nxt[:dh, :dw] = False
        elif corner == "tr":
            nxt[:dh, w - dw :] = False
        elif corner == "bl":
            nxt[h - dh :, :dw] = False
        else:
            nxt[h - dh :, w - dw :] = False
    else:
        side = rng.choice(("top", "bottom", "left", "right"))
        if side in ("top", "bottom"):
            if w <= 3:
                return None
            depth = rng.randint(1, min(2, h - 2))
            width = rng.randint(1, w - 2)
            start = rng.randint(1, w - width - 1)
            rows = slice(0, depth) if side == "top" else slice(h - depth, h)
            nxt[rows, start : start + width] = False
        else:
            if h <= 3:
                return None
            depth = rng.randint(1, min(2, w - 2))
            height = rng.randint(1, h - 2)
            start = rng.randint(1, h - height - 1)
            cols = slice(0, depth) if side == "left" else slice(w - depth, w)
            nxt[start : start + height, cols] = False

    if not nxt.any():
        return None
    if not _is_connected(nxt):
        return None
    if not _interior_cells(nxt):
        return None
    return nxt


def _generate_full_shape(
    rng: random.Random,
    pair_count: int,
    scale: int,
    used_keys: set[tuple[tuple[int, int], bytes]],
) -> np.ndarray | None:
    if scale == 3:
        max_dim = 5 if pair_count == 2 else 4
    else:
        max_dim = 6 if pair_count == 2 else 5

    for _ in range(120):
        h = rng.randint(3, max_dim)
        w = rng.randint(3, max_dim)
        mask = np.ones((h, w), dtype=bool)

        cut_tries = rng.randint(0, 3)
        for _ in range(cut_tries):
            candidate = _try_boundary_cut(mask, rng)
            if candidate is not None:
                mask = candidate

        key = _mask_key(mask)
        if key in used_keys:
            continue
        if len(_interior_cells(mask)) == 0:
            continue
        used_keys.add(key)
        return mask
    return None


def _generate_small_variant(rng: random.Random, full_mask: np.ndarray) -> np.ndarray | None:
    interior = _interior_cells(full_mask)
    if not interior:
        return None

    for _ in range(100):
        rng.shuffle(interior)
        max_holes = min(3, len(interior))
        hole_count = rng.randint(1, max_holes)
        removed: list[tuple[int, int]] = []
        for cell in interior:
            if any(abs(cell[0] - rr) + abs(cell[1] - cc) <= 1 for rr, cc in removed):
                continue
            removed.append(cell)
            if len(removed) == hole_count:
                break
        if len(removed) != hole_count:
            continue

        small = full_mask.copy()
        for r, c in removed:
            small[r, c] = False
        if _is_connected(small):
            return small
    return None


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    patch = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
    patch[mask] = color


def _place_objects(
    rng: random.Random,
    objects: list[dict],
) -> bool:
    occupancy = np.zeros((SCRATCH_SIZE, SCRATCH_SIZE), dtype=bool)
    placements = sorted(objects, key=lambda obj: (-int(obj["mask"].sum()), -obj["mask"].shape[0] * obj["mask"].shape[1]))
    for obj in placements:
        h, w = obj["mask"].shape
        placed = False
        for _ in range(250):
            top = rng.randint(1, SCRATCH_SIZE - h - 1)
            left = rng.randint(1, SCRATCH_SIZE - w - 1)
            r0 = top - 1
            c0 = left - 1
            r1 = top + h + 1
            c1 = left + w + 1
            if occupancy[r0:r1, c0:c1].any():
                continue
            occupancy[r0:r1, c0:c1] = True
            obj["top"] = top
            obj["left"] = left
            placed = True
            break
        if not placed:
            return False
    return True


def _build_pair_data(rng: random.Random) -> list[dict] | None:
    pair_count = rng.choice((2, 3))
    scales = [rng.choice((2, 3)) for _ in range(pair_count)]
    used_keys: set[tuple[tuple[int, int], bytes]] = set()
    pairs: list[dict] = []

    for scale in scales:
        full = _generate_full_shape(rng, pair_count, scale, used_keys)
        if full is None:
            return None
        small = _generate_small_variant(rng, full)
        if small is None:
            return None
        big = np.kron(full.astype(int), np.ones((scale, scale), dtype=int)).astype(bool)
        pairs.append(
            {
                "scale": scale,
                "full_mask": full,
                "small_mask": small,
                "big_mask": big,
            }
        )
    return pairs


def _render_example(rng: random.Random, pairs: list[dict]) -> tuple[np.ndarray, np.ndarray] | None:
    colors = rng.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], 2 * len(pairs))
    objects: list[dict] = []

    for idx, pair in enumerate(pairs):
        small_color = colors[2 * idx]
        big_color = colors[2 * idx + 1]
        pair["small_color"] = small_color
        pair["big_color"] = big_color
        objects.append({"pair": idx, "role": "small", "mask": pair["small_mask"]})
        objects.append({"pair": idx, "role": "big", "mask": pair["big_mask"]})

    if not _place_objects(rng, objects):
        return None

    for obj in objects:
        pair = pairs[obj["pair"]]
        if obj["role"] == "small":
            pair["small_top"] = obj["top"]
            pair["small_left"] = obj["left"]
        else:
            pair["big_top"] = obj["top"]
            pair["big_left"] = obj["left"]

    full_input = np.zeros((SCRATCH_SIZE, SCRATCH_SIZE), dtype=int)
    full_output = np.zeros((SCRATCH_SIZE, SCRATCH_SIZE), dtype=int)

    for pair in pairs:
        _paint_mask(full_input, pair["small_mask"], pair["small_top"], pair["small_left"], pair["small_color"])
        _paint_mask(full_input, pair["big_mask"], pair["big_top"], pair["big_left"], pair["big_color"])

        _paint_mask(full_output, pair["small_mask"], pair["small_top"], pair["small_left"], pair["big_color"])
        _paint_mask(full_output, pair["big_mask"], pair["big_top"], pair["big_left"], pair["big_color"])
        missing = pair["full_mask"] & ~pair["small_mask"]
        scale = int(pair["scale"])
        for rr, cc in np.argwhere(missing):
            r = int(rr)
            c = int(cc)
            full_output[
                pair["big_top"] + r * scale : pair["big_top"] + (r + 1) * scale,
                pair["big_left"] + c * scale : pair["big_left"] + (c + 1) * scale,
            ] = BG

    filled = np.argwhere(full_input != BG)
    if filled.size == 0:
        return None
    r0, c0 = filled.min(axis=0)
    r1, c1 = filled.max(axis=0)

    top_pad = rng.randint(1, min(3, int(r0)))
    left_pad = rng.randint(1, min(3, int(c0)))
    bottom_pad = rng.randint(1, min(3, SCRATCH_SIZE - 1 - int(r1)))
    right_pad = rng.randint(1, min(3, SCRATCH_SIZE - 1 - int(c1)))

    top = int(r0 - top_pad)
    left = int(c0 - left_pad)
    bottom = int(r1 + bottom_pad + 1)
    right = int(c1 + right_pad + 1)

    inp = full_input[top:bottom, left:right].copy()
    out = full_output[top:bottom, left:right].copy()
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = random.Random()
        pairs = _build_pair_data(rng)
        if pairs is None:
            continue
        rendered = _render_example(rng, pairs)
        if rendered is None:
            continue
        inp, expected = rendered
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("generate_input failed to construct a valid puzzle instance")
