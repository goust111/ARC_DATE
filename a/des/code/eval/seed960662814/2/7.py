from __future__ import annotations

import numpy as np


FRAME_COLOR = 8
FRAGMENT_COLOR = 5
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _tight_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    tight = _crop_mask(np.asarray(mask, dtype=bool))
    return tuple(map(int, tight.shape)), tight.tobytes()


def _apply_flip(mask: np.ndarray, flip_kind: str) -> np.ndarray:
    if flip_kind == "lr":
        return np.fliplr(mask)
    if flip_kind == "ud":
        return np.flipud(mask)
    raise AssertionError(f"unknown flip_kind: {flip_kind}")


def _side_to_flip(side: str) -> str:
    if side in {"top", "bottom"}:
        return "lr"
    if side in {"left", "right"}:
        return "ud"
    raise AssertionError(f"unknown side: {side}")


def _marked_side(patch: np.ndarray) -> str | None:
    sides = {
        "top": patch[0, :],
        "bottom": patch[-1, :],
        "left": patch[:, 0],
        "right": patch[:, -1],
    }
    marked: list[str] = []
    for side, line in sides.items():
        idx = np.flatnonzero(line != FRAME_COLOR)
        if idx.size == 0:
            continue
        if idx[-1] - idx[0] + 1 != idx.size:
            return None
        if len(set(map(int, line[idx]))) != 1:
            return None
        marked.append(side)
    if len(marked) != 1:
        return None
    return marked[0]


def _detect_containers(grid: np.ndarray, bg: int) -> list[dict]:
    comps = _components((grid != bg) & (grid != FRAGMENT_COLOR))
    containers: list[dict] = []

    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
            continue

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border_mask = np.zeros_like(patch, dtype=bool)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True
        if np.any(patch[border_mask] == bg):
            continue

        side = _marked_side(patch)
        if side is None:
            continue

        clean_border = patch[border_mask].copy()
        clean_border[clean_border != FRAME_COLOR] = FRAME_COLOR
        if not np.all(clean_border == FRAME_COLOR):
            continue

        interior = patch[1:-1, 1:-1]
        if interior.size == 0:
            continue
        hole_mask_full = interior == bg
        if not hole_mask_full.any():
            continue
        fill_values = np.unique(interior[~hole_mask_full])
        if fill_values.size != 1:
            continue
        fill_color = int(fill_values[0])
        if fill_color in {bg, FRAME_COLOR, FRAGMENT_COLOR}:
            continue

        hole_coords = [tuple(map(int, xy)) for xy in np.argwhere(hole_mask_full)]
        tight_hole = _tight_mask(hole_coords)
        flip_kind = _side_to_flip(side)
        containers.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "fill_color": fill_color,
                "hole_mask_full": hole_mask_full.copy(),
                "fragment_key": _mask_key(_apply_flip(tight_hole, flip_kind)),
            }
        )

    assert containers, "expected at least one container"
    return containers


def _detect_fragments(grid: np.ndarray) -> list[dict]:
    comps = _components(grid == FRAGMENT_COLOR)
    fragments: list[dict] = []
    for comp in comps:
        mask = _tight_mask(comp)
        fragments.append({"coords": comp, "key": _mask_key(mask)})
    assert fragments, "expected at least one fragment"
    return fragments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each container's missing interior with the uniquely matching grey fragment,
    after interpreting the border band as a left-right or up-down mirror instruction.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    containers = _detect_containers(g, bg)
    fragments = _detect_fragments(g)
    assert len(containers) == len(fragments), "fragment count must match container count"

    fragment_map: dict[tuple[tuple[int, int], bytes], dict] = {}
    for fragment in fragments:
        assert fragment["key"] not in fragment_map, "fragment shapes must be unique"
        fragment_map[fragment["key"]] = fragment

    out = g.copy()
    used_keys: set[tuple[tuple[int, int], bytes]] = set()

    for container in containers:
        key = container["fragment_key"]
        assert key in fragment_map, "container hole has no matching fragment"
        assert key not in used_keys, "fragment match must be one-to-one"
        used_keys.add(key)

        r0, c0, r1, c1 = container["bbox"]
        hole_mask_full = container["hole_mask_full"]
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        inner[hole_mask_full] = int(container["fill_color"])

    for fragment in fragments:
        for r, c in fragment["coords"]:
            out[r, c] = bg

    return out


# --------------- generator ---------------

GRID_H_RANGE = (18, 31)
GRID_W_RANGE = (18, 31)
CONTAINER_COUNT_CHOICES = (2, 3)
INTERIOR_H_RANGE = (4, 8)
INTERIOR_W_RANGE = (4, 8)
HOLE_AREA_MIN = 3
MAX_GEN_ATTEMPTS = 400
PLACEMENT_PAD = 1


def _random_connected_mask(rng: np.random.Generator, h: int, w: int, area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    seed = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    mask[seed] = True
    while int(mask.sum()) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in np.argwhere(mask):
            rr = int(r)
            cc = int(c)
            for dr, dc in DIRS4:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and not bool(mask[nr, nc]):
                    frontier.append((int(nr), int(nc)))
        pick = frontier[int(rng.integers(0, len(frontier)))]
        mask[pick] = True
    return _crop_mask(mask)


def _make_hole_mask(
    rng: np.random.Generator, ih: int, iw: int, flip_kind: str
) -> np.ndarray | None:
    max_area = max(HOLE_AREA_MIN, (ih * iw * 2) // 5)
    for _ in range(120):
        raw_h = int(rng.integers(2, min(ih, 5) + 1))
        raw_w = int(rng.integers(2, min(iw, 5) + 1))
        max_raw_area = min(raw_h * raw_w - 1, max_area)
        if max_raw_area < HOLE_AREA_MIN:
            continue
        area = int(rng.integers(HOLE_AREA_MIN, max_raw_area + 1))
        mask = _random_connected_mask(rng, raw_h, raw_w, area)
        if mask.shape[0] > ih or mask.shape[1] > iw:
            continue
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if int(mask.sum()) == int(mask.shape[0] * mask.shape[1]):
            continue
        if int(mask.sum()) >= ih * iw:
            continue
        if np.array_equal(mask, _apply_flip(mask, flip_kind)):
            continue
        return mask
    return None


def _band_segment(rng: np.random.Generator, length: int) -> tuple[int, int]:
    band_len = int(rng.integers(2, length - 1))
    start = int(rng.integers(1, length - band_len))
    return start, start + band_len


def _make_container(
    rng: np.random.Generator,
    bg: int,
    fill_color: int,
    band_color: int,
    side: str,
    used_fragment_keys: set[tuple[tuple[int, int], bytes]],
) -> dict | None:
    flip_kind = _side_to_flip(side)
    for _ in range(120):
        ih = int(rng.integers(*INTERIOR_H_RANGE))
        iw = int(rng.integers(*INTERIOR_W_RANGE))
        hole_mask = _make_hole_mask(rng, ih, iw, flip_kind)
        if hole_mask is None:
            continue
        hh, hw = map(int, hole_mask.shape)
        off_r = int(rng.integers(0, ih - hh + 1))
        off_c = int(rng.integers(0, iw - hw + 1))

        input_patch = np.full((ih + 2, iw + 2), fill_color, dtype=int)
        input_patch[0, :] = FRAME_COLOR
        input_patch[-1, :] = FRAME_COLOR
        input_patch[:, 0] = FRAME_COLOR
        input_patch[:, -1] = FRAME_COLOR

        out_patch = input_patch.copy()
        hole_slice = input_patch[1 + off_r : 1 + off_r + hh, 1 + off_c : 1 + off_c + hw]
        hole_slice[hole_mask] = bg

        line_len = iw + 2 if side in {"top", "bottom"} else ih + 2
        start, end = _band_segment(rng, line_len)
        if side == "top":
            input_patch[0, start:end] = band_color
            out_patch[0, start:end] = band_color
        elif side == "bottom":
            input_patch[-1, start:end] = band_color
            out_patch[-1, start:end] = band_color
        elif side == "left":
            input_patch[start:end, 0] = band_color
            out_patch[start:end, 0] = band_color
        else:
            input_patch[start:end, -1] = band_color
            out_patch[start:end, -1] = band_color

        fragment_mask = _apply_flip(hole_mask, flip_kind)
        fragment_key = _mask_key(fragment_mask)
        if fragment_key in used_fragment_keys:
            continue

        fragment_patch = np.full(fragment_mask.shape, bg, dtype=int)
        fragment_patch[fragment_mask] = FRAGMENT_COLOR
        used_fragment_keys.add(fragment_key)
        return {"input_patch": input_patch, "output_patch": out_patch, "fragment_patch": fragment_patch}
    return None


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int, *, transparent: int | None = None) -> None:
    h, w = map(int, patch.shape)
    view = dst[top : top + h, left : left + w]
    if transparent is None:
        view[:] = patch
    else:
        mask = patch != transparent
        view[mask] = patch[mask]


def _place_objects(
    rng: np.random.Generator, grid_shape: tuple[int, int], objects: list[dict]
) -> bool:
    h, w = map(int, grid_shape)
    blocked = np.zeros((h, w), dtype=bool)
    pieces: list[tuple[int, str]] = []
    for idx, obj in enumerate(objects):
        pieces.append((idx, "input_patch"))
        pieces.append((idx, "fragment_patch"))
    pieces.sort(
        key=lambda x: int(objects[x[0]][x[1]].shape[0]) * int(objects[x[0]][x[1]].shape[1]),
        reverse=True,
    )

    for idx, patch_name in pieces:
        patch = objects[idx][patch_name]
        ph, pw = map(int, patch.shape)
        candidates: list[tuple[int, int]] = []
        for r0 in range(h - ph + 1):
            for c0 in range(w - pw + 1):
                if blocked[r0 : r0 + ph, c0 : c0 + pw].any():
                    continue
                candidates.append((int(r0), int(c0)))
        if not candidates:
            return False
        pick = candidates[int(rng.integers(0, len(candidates)))]
        objects[idx][f"{patch_name}_pos"] = pick
        blocked[
            max(0, pick[0] - PLACEMENT_PAD) : min(h, pick[0] + ph + PLACEMENT_PAD),
            max(0, pick[1] - PLACEMENT_PAD) : min(w, pick[1] + pw + PLACEMENT_PAD),
        ] = True
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    bg_choices = [c for c in range(10) if c not in {FRAME_COLOR, FRAGMENT_COLOR}]
    bg = int(rng.choice(bg_choices))
    n_containers = int(rng.choice(CONTAINER_COUNT_CHOICES))

    usable = [c for c in range(10) if c not in {bg, FRAME_COLOR, FRAGMENT_COLOR}]
    if len(usable) < 2 * n_containers:
        return None
    rng.shuffle(usable)
    fill_colors = usable[:n_containers]
    band_colors = usable[n_containers : 2 * n_containers]

    used_fragment_keys: set[tuple[tuple[int, int], bytes]] = set()
    objects: list[dict] = []
    sides = ["top", "bottom", "left", "right"]
    for i in range(n_containers):
        side = sides[int(rng.integers(0, len(sides)))]
        obj = _make_container(
            rng,
            bg=bg,
            fill_color=int(fill_colors[i]),
            band_color=int(band_colors[i]),
            side=side,
            used_fragment_keys=used_fragment_keys,
        )
        if obj is None:
            return None
        objects.append(obj)

    grid_h = int(rng.integers(*GRID_H_RANGE))
    grid_w = int(rng.integers(*GRID_W_RANGE))
    if not _place_objects(rng, (grid_h, grid_w), objects):
        return None

    inp = np.full((grid_h, grid_w), bg, dtype=int)
    out = np.full((grid_h, grid_w), bg, dtype=int)
    for obj in objects:
        in_r, in_c = obj["input_patch_pos"]
        fr_r, fr_c = obj["fragment_patch_pos"]
        _paste(inp, obj["input_patch"], in_r, in_c)
        _paste(inp, obj["fragment_patch"], fr_r, fr_c, transparent=bg)
        _paste(out, obj["output_patch"], in_r, in_c)

    try:
        solved = main(inp)
    except Exception:
        return None
    if np.array_equal(solved, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        rng = np.random.default_rng()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
