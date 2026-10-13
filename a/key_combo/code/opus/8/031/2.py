from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
TEMPLATE = 5
NOISE = 7
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SPRITE_COLORS = [1, 2, 3, 4, 6, 8, 9]
ALLOWED_DIMS = [(2, 3), (3, 2), (2, 4), (4, 2), (3, 3), (2, 5), (5, 2), (3, 4), (4, 3)]
NOISY_TARGETS = [(3, 3), (3, 4), (4, 3)]
MAX_GEN_ATTEMPTS = 400


def neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def bbox_from_coords(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def bbox_shape(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox_from_coords(coords)
    return r1 - r0 + 1, c1 - c0 + 1


def normalize_coords(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0, c0, _, _ = bbox_from_coords(coords)
    return {(r - r0, c - c0) for r, c in coords}


def component_coords(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    if connectivity == 4:
        dirs = DIR4
    else:
        dirs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    q = deque([next(iter(cells))])
    seen = {q[0]}
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(cells)


def count_neighbors(cell: tuple[int, int], cells: set[tuple[int, int]]) -> int:
    r, c = cell
    return sum((nr, nc) in cells for nr, nc in neighbors4(r, c))


def largest_component_of_color(grid: np.ndarray, color: int) -> list[tuple[int, int]]:
    comps = component_coords(grid == color, connectivity=4)
    assert comps
    comps.sort(key=len, reverse=True)
    return comps[0]


def hole_specs_from_template(crop: np.ndarray) -> list[dict]:
    holes: list[dict] = []
    for comp in component_coords(crop == BG, connectivity=4):
        r0, c0, r1, c1 = bbox_from_coords(comp)
        if r0 == 0 or c0 == 0 or r1 == crop.shape[0] - 1 or c1 == crop.shape[1] - 1:
            continue
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            mask[r - r0, c - c0] = True
        assert mask.all()
        holes.append({"r0": r0, "c0": c0, "h": mask.shape[0], "w": mask.shape[1]})
    return holes


def cleaned_sprite_patch(
    body_coords: set[tuple[int, int]],
    merge_noise: set[tuple[int, int]],
    color: int,
) -> tuple[np.ndarray, np.ndarray]:
    cleaned = body_coords | merge_noise
    r0, c0, r1, c1 = bbox_from_coords(cleaned)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    mask = np.zeros_like(patch, dtype=bool)
    for r, c in cleaned:
        patch[r - r0, c - c0] = color
        mask[r - r0, c - c0] = True
    return patch, mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Clean each external sprite's orange noise so its true bounding box matches one rectangular cavity, then insert it into the template.

    Input structure:
    - Black background surrounds one large grey template component containing several internal black rectangular cavities.
    - Outside the template sit 3-4 small monochrome 4-connected sprites in distinct colors.
    - Each sprite has adjacent orange noise pixels; some orange pixels weakly touch the sprite while others touch it from two or more sides.

    Rule:
    - Find the largest grey connected component and treat its bounding box as the template crop; every enclosed black component in that crop is a cavity.
    - For each non-grey, non-orange external sprite, collect the orange pixels 4-adjacent to it and clean them individually: 4-neighbor count >= 2 to the sprite means merge into the sprite color, otherwise erase to background.
    - Recompute each cleaned sprite's bounding box, match it to the unique cavity with the same height and width, and paste the cleaned sprite with bbox top-left aligned to the cavity top-left.
    - Return only the template bounding box region after filling the cavities.

    Key insight:
    - The orange pixels are not decorative: deleting weak contacts removes fake bbox inflation, while merging strong contacts restores missing sprite cells, and only the cleaned bbox yields a unique cavity match.
    """
    g = np.asarray(grid, dtype=int)
    template_comp = largest_component_of_color(g, TEMPLATE)
    tr0, tc0, tr1, tc1 = bbox_from_coords(template_comp)
    crop = g[tr0 : tr1 + 1, tc0 : tc1 + 1].copy()
    holes = hole_specs_from_template(crop)
    holes_by_dim: dict[tuple[int, int], dict] = {}
    for hole in holes:
        dim = (int(hole["h"]), int(hole["w"]))
        assert dim not in holes_by_dim
        holes_by_dim[dim] = hole

    H, W = g.shape
    sprite_comps: list[dict] = []
    for color in sorted(set(int(x) for x in g.ravel()) - {BG, TEMPLATE, NOISE}):
        body_mask = g == color
        for comp in component_coords(body_mask, connectivity=4):
            if any(tr0 <= r <= tr1 and tc0 <= c <= tc1 for r, c in comp):
                continue
            sprite_comps.append({"color": color, "body": set(comp)})

    orange_owners: dict[tuple[int, int], set[int]] = {}
    for idx, sprite in enumerate(sprite_comps):
        for r, c in sprite["body"]:
            for nr, nc in neighbors4(r, c):
                if in_bounds((H, W), nr, nc) and g[nr, nc] == NOISE:
                    orange_owners.setdefault((nr, nc), set()).add(idx)

    out = crop.copy()
    for idx, sprite in enumerate(sprite_comps):
        body = sprite["body"]
        color = int(sprite["color"])
        merge_noise: set[tuple[int, int]] = set()
        for cell, owners in orange_owners.items():
            if owners != {idx}:
                continue
            if count_neighbors(cell, body) >= 2:
                merge_noise.add(cell)
        patch, mask = cleaned_sprite_patch(body, merge_noise, color)
        dim = patch.shape
        assert dim in holes_by_dim
        hole = holes_by_dim[dim]
        rr0 = int(hole["r0"])
        cc0 = int(hole["c0"])
        out_slice = out[rr0 : rr0 + dim[0], cc0 : cc0 + dim[1]]
        out_slice[mask] = patch[mask]

    return out


def random_connected_mask(h: int, w: int, require_merge: bool) -> set[tuple[int, int]] | None:
    area_lo = max(4, max(h, w))
    area_hi = min(8, h * w - 1)
    if require_merge:
        area_lo = max(area_lo, 5)
    if area_hi < area_lo:
        return None
    for _ in range(200):
        area = random.randint(area_lo, area_hi)
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue
        if bbox_shape(cells) != (h, w):
            continue
        if len(cells) == h * w:
            continue
        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        if 0 not in rows or h - 1 not in rows or 0 not in cols or w - 1 not in cols:
            continue
        if require_merge and not merge_candidates(cells):
            continue
        return cells
    return None


def merge_candidates(cleaned: set[tuple[int, int]]) -> list[tuple[int, int]]:
    candidates = []
    for cell in cleaned:
        if count_neighbors(cell, cleaned) < 2:
            continue
        body = cleaned - {cell}
        if len(body) < 4:
            continue
        if is_connected(body):
            candidates.append(cell)
    return candidates


def choose_delete_noise(body: set[tuple[int, int]], h: int, w: int, side: str) -> tuple[int, int] | None:
    if side == "top":
        cols = [c for r, c in body if r == 0]
        return None if not cols else (-1, random.choice(cols))
    if side == "bottom":
        cols = [c for r, c in body if r == h - 1]
        return None if not cols else (h, random.choice(cols))
    if side == "left":
        rows = [r for r, c in body if c == 0]
        return None if not rows else (random.choice(rows), -1)
    rows = [r for r, c in body if c == w - 1]
    return None if not rows else (random.choice(rows), w)


def make_input_patch(
    body: set[tuple[int, int]],
    merge_noise: set[tuple[int, int]],
    delete_noise: set[tuple[int, int]],
    color: int,
) -> dict:
    all_cells = body | merge_noise | delete_noise
    r0, c0, r1, c1 = bbox_from_coords(all_cells)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    nonbg = np.zeros_like(patch, dtype=bool)
    for r, c in body:
        patch[r - r0, c - c0] = color
        nonbg[r - r0, c - c0] = True
    for r, c in merge_noise | delete_noise:
        patch[r - r0, c - c0] = NOISE
        nonbg[r - r0, c - c0] = True
    return {"patch": patch, "mask": nonbg, "bbox_dim": patch.shape}


def try_place_holes(template_h: int, template_w: int, dims: list[tuple[int, int]]) -> list[dict] | None:
    order = sorted(enumerate(dims), key=lambda x: x[1][0] * x[1][1], reverse=True)
    occupied = np.zeros((template_h, template_w), dtype=bool)
    placed: list[dict | None] = [None] * len(dims)
    for idx, (h, w) in order:
        choices = []
        for r0 in range(1, template_h - h):
            for c0 in range(1, template_w - w):
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(template_h, r0 + h + 1)
                cc1 = min(template_w, c0 + w + 1)
                if occupied[rr0:rr1, cc0:cc1].any():
                    continue
                choices.append((r0, c0))
        if not choices:
            return None
        r0, c0 = random.choice(choices)
        occupied[r0 : r0 + h, c0 : c0 + w] = True
        placed[idx] = {"r0": r0, "c0": c0, "h": h, "w": w}
    return [p for p in placed if p is not None]


def place_patch_on_grid(
    grid: np.ndarray,
    patch: np.ndarray,
    r0: int,
    c0: int,
) -> None:
    nonzero = patch != BG
    grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][nonzero] = patch[nonzero]


def try_place_sprites(
    H: int,
    W: int,
    template_bbox: tuple[int, int, int, int],
    patches: list[dict],
) -> list[tuple[int, int]] | None:
    tr0, tc0, tr1, tc1 = template_bbox
    reserved = np.zeros((H, W), dtype=bool)
    rr0 = max(0, tr0 - 1)
    cc0 = max(0, tc0 - 1)
    rr1 = min(H, tr1 + 2)
    cc1 = min(W, tc1 + 2)
    reserved[rr0:rr1, cc0:cc1] = True
    placed: list[tuple[int, int] | None] = [None] * len(patches)
    order = sorted(enumerate(patches), key=lambda x: x[1]["patch"].shape[0] * x[1]["patch"].shape[1], reverse=True)
    for idx, spec in order:
        ph, pw = spec["patch"].shape
        choices = []
        for r0 in range(H - ph + 1):
            for c0 in range(W - pw + 1):
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(H, r0 + ph + 1)
                cc1 = min(W, c0 + pw + 1)
                if reserved[rr0:rr1, cc0:cc1].any():
                    continue
                choices.append((r0, c0))
        if not choices:
            return None
        r0, c0 = random.choice(choices)
        placed[idx] = (r0, c0)
        reserved[max(0, r0 - 1) : min(H, r0 + ph + 1), max(0, c0 - 1) : min(W, c0 + pw + 1)] = True
    return [p for p in placed if p is not None]


def output_if_delete_all_orange(meta: dict) -> np.ndarray:
    out = meta["template_crop"].copy()
    for sprite, hole in zip(meta["sprites"], meta["holes"]):
        body = sprite["body"]
        color = int(sprite["color"])
        r0, c0, r1, c1 = bbox_from_coords(body)
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        mask = np.zeros_like(patch, dtype=bool)
        for r, c in body:
            patch[r - r0, c - c0] = color
            mask[r - r0, c - c0] = True
        rr0 = int(hole["r0"])
        cc0 = int(hole["c0"])
        out[rr0 : rr0 + patch.shape[0], cc0 : cc0 + patch.shape[1]][mask] = patch[mask]
    return out


def validate_case(inp: np.ndarray, out: np.ndarray, meta: dict) -> None:
    assert inp.ndim == 2 and out.ndim == 2
    assert max(inp.shape) <= 30 and max(out.shape) <= 30
    assert np.array_equal(main(inp), out)
    hole_dims = [(h["h"], h["w"]) for h in meta["holes"]]
    assert len(hole_dims) == len(set(hole_dims))
    cleaned_dims = [bbox_shape(s["body"] | s["merge_noise"]) for s in meta["sprites"]]
    assert Counter(cleaned_dims) == Counter(hole_dims)
    noisy_dims = [bbox_shape(s["body"] | s["merge_noise"] | s["delete_noise"]) for s in meta["sprites"]]
    assert Counter(noisy_dims).most_common(1)[0][1] >= 2
    assert any(s["merge_noise"] for s in meta["sprites"])
    assert any(s["delete_noise"] for s in meta["sprites"])
    assert not np.array_equal(output_if_delete_all_orange(meta), out)


def build_case() -> tuple[np.ndarray, np.ndarray]:
    noisy_target = random.choice(NOISY_TARGETS)
    dims = [(noisy_target[0] - 1, noisy_target[1]), (noisy_target[0], noisy_target[1] - 1)]
    count = random.choice([3, 4])
    extra_pool = [d for d in ALLOWED_DIMS if d not in dims]
    dims.extend(random.sample(extra_pool, count - 2))
    random.shuffle(dims)

    noisy_pair = [
        i
        for i, dim in enumerate(dims)
        if dim in {(noisy_target[0] - 1, noisy_target[1]), (noisy_target[0], noisy_target[1] - 1)}
    ]
    assert len(noisy_pair) == 2
    extra_indices = [i for i in range(count) if i not in noisy_pair]
    merge_idx = random.choice(extra_indices)

    colors = random.sample(SPRITE_COLORS, count)
    sprites = []
    for i, ((h, w), color) in enumerate(zip(dims, colors)):
        cleaned = random_connected_mask(h, w, require_merge=(i == merge_idx))
        if cleaned is None:
            raise ValueError("shape generation failed")
        merge_noise: set[tuple[int, int]] = set()
        body = set(cleaned)
        if i == merge_idx:
            merge_cell = random.choice(merge_candidates(cleaned))
            body = cleaned - {merge_cell}
            merge_noise = {merge_cell}
        delete_noise: set[tuple[int, int]] = set()
        if (h, w) == (noisy_target[0] - 1, noisy_target[1]):
            side = random.choice(["top", "bottom"])
            cell = choose_delete_noise(body, h, w, side)
            if cell is None:
                raise ValueError("missing delete noise cell")
            delete_noise.add(cell)
        elif (h, w) == (noisy_target[0], noisy_target[1] - 1):
            side = random.choice(["left", "right"])
            cell = choose_delete_noise(body, h, w, side)
            if cell is None:
                raise ValueError("missing delete noise cell")
            delete_noise.add(cell)
        else:
            side = random.choice(["top", "bottom", "left", "right"])
            cell = choose_delete_noise(body, h, w, side)
            if cell is None:
                raise ValueError("missing extra noise cell")
            delete_noise.add(cell)
        sprites.append(
            {
                "color": color,
                "body": body,
                "merge_noise": merge_noise,
                "delete_noise": delete_noise,
                "cleaned_dim": (h, w),
            }
        )

    hole_dims = [s["cleaned_dim"] for s in sprites]
    for _ in range(80):
        template_h = random.randint(max(10, max(h for h, _ in hole_dims) + 5), 16)
        template_w = random.randint(max(11, max(w for _, w in hole_dims) + 6), 18)
        holes = try_place_holes(template_h, template_w, hole_dims)
        if holes is not None:
            break
    else:
        raise ValueError("hole placement failed")

    template_crop = np.full((template_h, template_w), TEMPLATE, dtype=int)
    for hole in holes:
        template_crop[hole["r0"] : hole["r0"] + hole["h"], hole["c0"] : hole["c0"] + hole["w"]] = BG

    out = template_crop.copy()
    for sprite, hole in zip(sprites, holes):
        patch, mask = cleaned_sprite_patch(sprite["body"], sprite["merge_noise"], int(sprite["color"]))
        out[hole["r0"] : hole["r0"] + patch.shape[0], hole["c0"] : hole["c0"] + patch.shape[1]][mask] = patch[mask]

    H = random.randint(max(20, template_h + 7), min(29, template_h + 12))
    W = random.randint(max(20, template_w + 7), min(29, template_w + 12))
    margin_r = H - template_h
    margin_c = W - template_w
    tr0 = random.randint(2, margin_r - 2)
    tc0 = random.randint(2, margin_c - 2)
    tr1 = tr0 + template_h - 1
    tc1 = tc0 + template_w - 1

    inp = np.zeros((H, W), dtype=int)
    inp[tr0 : tr1 + 1, tc0 : tc1 + 1] = template_crop

    patch_specs = []
    for sprite in sprites:
        patch_specs.append(
            make_input_patch(
                sprite["body"],
                sprite["merge_noise"],
                sprite["delete_noise"],
                int(sprite["color"]),
            )
        )
    placements = try_place_sprites(inp.shape[0], inp.shape[1], (tr0, tc0, tr1, tc1), patch_specs)
    if placements is None:
        raise ValueError("sprite placement failed")
    for patch_spec, (r0, c0) in zip(patch_specs, placements):
        place_patch_on_grid(inp, patch_spec["patch"], r0, c0)

    meta = {"template_crop": template_crop, "holes": holes, "sprites": sprites}
    validate_case(inp, out, meta)
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            return build_case()[0]
        except ValueError:
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
