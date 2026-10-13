from __future__ import annotations

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
RELATIONS = ("horizontal", "vertical", "rotation")
MAX_GENERATION_ATTEMPTS = 400
_RNG = np.random.default_rng()


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _coords_to_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    mask[arr[:, 0] - r0, arr[:, 1] - c0] = True
    return mask, (int(r0), int(c0), int(r1), int(c1))


def _transform_mask(mask: np.ndarray, relation: str) -> np.ndarray:
    if relation == "horizontal":
        return np.fliplr(mask)
    if relation == "vertical":
        return np.flipud(mask)
    if relation == "rotation":
        return np.rot90(mask, 2)
    raise AssertionError(f"unknown relation: {relation}")


def _relation_candidates(base_mask: np.ndarray, pair_mask: np.ndarray) -> list[str]:
    if base_mask.shape != pair_mask.shape:
        return []
    candidates = []
    for relation in RELATIONS:
        if np.array_equal(pair_mask, _transform_mask(base_mask, relation)):
            candidates.append(relation)
    return candidates


def _assemble_full_mask(base_mask: np.ndarray, relation: str) -> tuple[np.ndarray, tuple[int, int]]:
    pair_mask = _transform_mask(base_mask, relation)
    h, w = map(int, base_mask.shape)

    if relation == "horizontal":
        full = np.zeros((h, 2 * w - 1), dtype=bool)
        full[:, :w] |= base_mask
        full[:, w - 1 :] |= pair_mask
        center = (h // 2, w - 1)
    elif relation == "vertical":
        full = np.zeros((2 * h - 1, w), dtype=bool)
        full[:h, :] |= base_mask
        full[h - 1 :, :] |= pair_mask
        center = (h - 1, w // 2)
    else:
        full = np.zeros((2 * h - 1, 2 * w - 1), dtype=bool)
        full[:h, :w] |= base_mask
        full[h - 1 :, w - 1 :] |= pair_mask
        center = (h - 1, w - 1)

    return full, center


def _paint_mask(out: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    coords = np.argwhere(mask)
    for r, c in coords:
        out[top + int(r), left + int(c)] = int(color)


def _find_markers(grid: np.ndarray) -> tuple[list[dict], np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    center_mask = np.zeros((H, W), dtype=bool)
    arm_mask = np.zeros((H, W), dtype=bool)
    markers = []

    for r in range(1, H - 1):
        for c in range(1, W - 1):
            color = int(g[r, c])
            if color in (BG, GREY):
                continue
            neighbors = ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
            if all(int(g[nr, nc]) == GREY for nr, nc in neighbors):
                center_mask[r, c] = True
                for nr, nc in neighbors:
                    arm_mask[nr, nc] = True
                markers.append({"color": color, "center": (int(r), int(c))})

    assert markers, "at least one anchor marker is required"
    assert int(arm_mask.sum()) == 4 * len(markers), "marker arms may not overlap"
    grey_mask = g == GREY
    assert np.array_equal(grey_mask, arm_mask), "grey pixels must be exactly marker arms"

    colors = [int(marker["color"]) for marker in markers]
    assert len(set(colors)) == len(colors), "marker colors must be unique"
    return markers, center_mask, arm_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored fragments, infer their symmetry relation, and rebuild the full shape at the matching colored cross marker.

    Input structure:
    - bg = black.
    - Each non-grey foreground color appears in exactly two scattered 4-connected fragments plus one cross-marker center of the same color.
    - Every marker is a 5-cell plus sign whose four arms are grey and whose center is recolored to the fragment-pair color.

    Rule:
    - Detect each colored cross center and use its color to match that anchor to the two fragments of the same color.
    - Choose the base fragment as the larger one, breaking ties by smaller top row; the other fragment must be the base's horizontal mirror, vertical mirror, or 180-degree rotation.
    - Reassemble the full symmetric figure from that relation and place it so the figure's center aligns with the marker center; remove the original fragments and the grey marker arms.

    Key insight:
    - The anchor color gives the pairing, but the output geometry comes from a second hidden cue: which symmetry transform links the two like-colored fragments.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    H, W = g.shape
    markers, center_mask, _ = _find_markers(g)

    output = np.full((H, W), BG, dtype=int)
    marker_by_color = {int(marker["color"]): marker for marker in markers}
    non_marker_colors = set(map(int, np.unique(g))) - {BG, GREY}
    assert non_marker_colors == set(marker_by_color), "every non-grey color must own one marker"

    for color, marker in marker_by_color.items():
        fragment_mask = (g == int(color)) & (~center_mask)
        components = _find_components(fragment_mask)
        assert len(components) == 2, "each color must have exactly two fragments"

        infos = []
        for coords in components:
            mask, bbox = _coords_to_mask(coords)
            infos.append(
                {
                    "coords": coords,
                    "mask": mask,
                    "bbox": bbox,
                    "area": int(mask.sum()),
                }
            )
        infos.sort(key=lambda item: (-item["area"], item["bbox"][0], item["bbox"][1], item["bbox"][3]))
        base = infos[0]
        pair = infos[1]

        candidates = _relation_candidates(base["mask"], pair["mask"])
        assert len(candidates) == 1, "fragment relation must be uniquely decodable"
        relation = candidates[0]

        full_mask, full_center = _assemble_full_mask(base["mask"], relation)
        mr, mc = marker["center"]
        cr, cc = full_center
        for r, c in np.argwhere(full_mask):
            rr = int(mr + r - cr)
            rc = int(mc + c - cc)
            assert 0 <= rr < H and 0 <= rc < W
            assert output[rr, rc] in (BG, color)
            output[rr, rc] = int(color)

    return output


def _grow_mask(h: int, w: int, target_area: int) -> np.ndarray:
    cells = {(0, 0)}
    r = 0
    c = 0
    while (r, c) != (h - 1, w - 1):
        options = []
        if r < h - 1:
            options.append((1, 0))
        if c < w - 1:
            options.append((0, 1))
        dr, dc = options[int(_RNG.integers(0, len(options)))]
        r += dr
        c += dc
        cells.add((r, c))

    while len(cells) < target_area:
        frontier = set()
        for rr, cc in cells:
            for dr, dc in DIR4:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            break
        pick = list(frontier)[int(_RNG.integers(0, len(frontier)))]
        cells.add(pick)

    mask = np.zeros((h, w), dtype=bool)
    for rr, cc in cells:
        mask[rr, cc] = True
    return mask


def _sample_base_mask(relation: str) -> np.ndarray | None:
    for _ in range(200):
        if relation == "horizontal":
            h = int(_RNG.choice(np.array([3, 5])))
            w = int(_RNG.integers(2, 5))
        elif relation == "vertical":
            h = int(_RNG.integers(2, 5))
            w = int(_RNG.choice(np.array([3, 5])))
        else:
            h = int(_RNG.choice(np.array([3, 5])))
            w = int(_RNG.choice(np.array([3, 5])))

        path_area = h + w - 1
        max_area = min(h * w - 1, path_area + 4)
        if max_area < path_area + 1:
            continue
        target_area = int(_RNG.integers(path_area + 1, max_area + 1))
        mask = _grow_mask(h, w, target_area)

        if relation == "horizontal" and h % 2 == 0:
            continue
        if relation == "vertical" and w % 2 == 0:
            continue
        if relation == "rotation" and (h % 2 == 0 or w % 2 == 0):
            continue
        if int(mask.sum()) >= h * w:
            continue
        if not mask[0, 0] or not mask[h - 1, w - 1]:
            continue
        if np.array_equal(mask, np.ones((h, w), dtype=bool)):
            continue
        if len(_relation_candidates(mask, _transform_mask(mask, relation))) != 1:
            continue

        full_mask, full_center = _assemble_full_mask(mask, relation)
        if not full_mask[full_center]:
            continue
        return mask
    return None


def _required_side(specs: list[dict]) -> int:
    footprint = 0
    max_dim = 0
    for spec in specs:
        fh, fw = spec["full_mask"].shape
        h, w = spec["base_mask"].shape
        footprint += (fh + 2) * (fw + 2) + 2 * (h + 2) * (w + 2)
        max_dim = max(max_dim, fh + 2, fw + 2, h + 2, w + 2)
    side = int(np.ceil(np.sqrt(footprint)))
    return max(max_dim, side)


def _can_place(blocked: np.ndarray, top: int, left: int, h: int, w: int, pad: int) -> bool:
    H, W = blocked.shape
    if top < 0 or left < 0 or top + h > H or left + w > W:
        return False
    r0 = max(0, top - pad)
    r1 = min(H, top + h + pad)
    c0 = max(0, left - pad)
    c1 = min(W, left + w + pad)
    return not bool(blocked[r0:r1, c0:c1].any())


def _mark_rect(blocked: np.ndarray, top: int, left: int, h: int, w: int, pad: int) -> None:
    H, W = blocked.shape
    r0 = max(0, top - pad)
    r1 = min(H, top + h + pad)
    c0 = max(0, left - pad)
    c1 = min(W, left + w + pad)
    blocked[r0:r1, c0:c1] = True


def _place_rect(blocked: np.ndarray, h: int, w: int, pad: int, attempts: int = 250) -> tuple[int, int] | None:
    H, W = blocked.shape
    if h > H or w > W:
        return None
    for _ in range(attempts):
        top = int(_RNG.integers(0, H - h + 1))
        left = int(_RNG.integers(0, W - w + 1))
        if _can_place(blocked, top, left, h, w, pad):
            _mark_rect(blocked, top, left, h, w, pad)
            return top, left
    return None


def _render_input(specs: list[dict], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for spec in specs:
        color = int(spec["color"])
        _paint_mask(grid, spec["base_mask"], spec["base_top"], spec["base_left"], color)
        _paint_mask(grid, spec["pair_mask"], spec["pair_top"], spec["pair_left"], color)
        mr, mc = spec["marker_center"]
        grid[mr, mc] = color
        grid[mr - 1, mc] = GREY
        grid[mr + 1, mc] = GREY
        grid[mr, mc - 1] = GREY
        grid[mr, mc + 1] = GREY
    return grid


def _render_expected_output(specs: list[dict], shape: tuple[int, int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for spec in specs:
        color = int(spec["color"])
        mr, mc = spec["marker_center"]
        cr, cc = spec["full_center"]
        for r, c in np.argwhere(spec["full_mask"]):
            rr = int(mr + r - cr)
            rc = int(mc + c - cc)
            assert out[rr, rc] in (BG, color)
            out[rr, rc] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    pair_count = int(_RNG.integers(2, 5))
    colors = _RNG.choice(np.array(COLOR_POOL), size=pair_count, replace=False)
    relations = [_RNG.choice(np.array(RELATIONS)).item() for _ in range(pair_count)]

    specs = []
    for color, relation in zip(colors, relations):
        base_mask = _sample_base_mask(relation)
        if base_mask is None:
            return None
        pair_mask = _transform_mask(base_mask, relation)
        full_mask, full_center = _assemble_full_mask(base_mask, relation)
        specs.append(
            {
                "color": int(color),
                "relation": relation,
                "base_mask": base_mask,
                "pair_mask": pair_mask,
                "full_mask": full_mask,
                "full_center": full_center,
            }
        )

    min_side = _required_side(specs)
    if min_side > 30:
        return None
    H = int(_RNG.integers(max(18, min_side), 31))
    W = int(_RNG.integers(max(18, min_side), 31))
    blocked = np.zeros((H, W), dtype=bool)

    anchor_order = sorted(range(pair_count), key=lambda i: specs[i]["full_mask"].size, reverse=True)
    for idx in anchor_order:
        spec = specs[idx]
        fh, fw = spec["full_mask"].shape
        pos = _place_rect(blocked, fh, fw, pad=1)
        if pos is None:
            return None
        top, left = pos
        cr, cc = spec["full_center"]
        spec["marker_center"] = (int(top + cr), int(left + cc))

    fragment_order = list(range(pair_count))
    _RNG.shuffle(fragment_order)
    for idx in fragment_order:
        spec = specs[idx]
        h, w = spec["base_mask"].shape
        base_pos = _place_rect(blocked, h, w, pad=1)
        if base_pos is None:
            return None
        pair_pos = _place_rect(blocked, h, w, pad=1)
        if pair_pos is None:
            return None
        spec["base_top"], spec["base_left"] = map(int, base_pos)
        spec["pair_top"], spec["pair_left"] = map(int, pair_pos)

    grid = _render_input(specs, (H, W))
    expected = _render_expected_output(specs, (H, W))
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
