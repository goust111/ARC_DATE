from __future__ import annotations

import numpy as np


BG = 0
SIDES = ("top", "bottom", "left", "right")

_RNG = np.random.default_rng()

MIN_FRAME_H = 4
MAX_FRAME_H = 9
MIN_FRAME_W = 4
MAX_FRAME_W = 10

GRID_H_BY_COLOR_COUNT = {
    3: (18, 28),
    4: (21, 32),
    5: (24, 36),
}
GRID_W_BY_COLOR_COUNT = {
    3: (20, 30),
    4: (24, 35),
    5: (28, 40),
}

PLACE_ATTEMPTS = 180
GEN_ATTEMPTS = 400


def _full_frame_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _full_frame_patch(h: int, w: int, color: int) -> np.ndarray:
    patch = np.full((h, w), BG, dtype=int)
    mask = _full_frame_mask(h, w)
    patch[mask] = int(color)
    return patch


def _missing_frame_patch(h: int, w: int, color: int, side: str) -> np.ndarray:
    patch = _full_frame_patch(h, w, color)
    if side == "top":
        patch[0, 1:-1] = BG
    elif side == "bottom":
        patch[-1, 1:-1] = BG
    elif side == "left":
        patch[1:-1, 0] = BG
    elif side == "right":
        patch[1:-1, -1] = BG
    else:
        raise ValueError(f"unknown side: {side}")
    return patch


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

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
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _component_info(coords: np.ndarray) -> dict[str, object]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    rel = coords - np.array([r0, c0], dtype=int)
    mask = np.zeros((h, w), dtype=bool)
    mask[rel[:, 0], rel[:, 1]] = True
    return {
        "bbox": (r0, c0, r1, c1),
        "mask": mask,
    }


def _frame_kind(mask: np.ndarray) -> str | None:
    h, w = mask.shape
    if h < 3 or w < 3:
        return None

    border = _full_frame_mask(h, w)
    if np.any(mask & ~border):
        return None
    if np.array_equal(mask, border):
        return "full"

    for side in SIDES:
        candidate = border.copy()
        if side == "top":
            candidate[0, 1:-1] = False
        elif side == "bottom":
            candidate[-1, 1:-1] = False
        elif side == "left":
            candidate[1:-1, 0] = False
        else:
            candidate[1:-1, -1] = False
        if np.array_equal(mask, candidate):
            return side
    return None


def _fill_missing_side(out: np.ndarray, color: int, bbox: tuple[int, int, int, int], side: str) -> None:
    r0, c0, r1, c1 = bbox
    if side == "top":
        out[r0, c0 : c1 + 1] = int(color)
    elif side == "bottom":
        out[r1, c0 : c1 + 1] = int(color)
    elif side == "left":
        out[r0 : r1 + 1, c0] = int(color)
    elif side == "right":
        out[r0 : r1 + 1, c1] = int(color)
    else:
        raise ValueError(f"unknown side: {side}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each same-color three-sided rectangular frame using its complete mate.

    The source edge is always monochrome, so copying-and-scaling that edge is
    equivalent to painting the whole missing side of the target frame.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    colors = [int(color) for color in np.unique(g) if int(color) != BG]
    for color in colors:
        components = [_component_info(coords) for coords in _find_components(g == color)]
        assert len(components) == 2, f"expected exactly two components for color {color}"

        full_count = 0
        target_bbox: tuple[int, int, int, int] | None = None
        missing_side: str | None = None

        for component in components:
            kind = _frame_kind(component["mask"])  # type: ignore[index]
            assert kind is not None, f"color {color} must form frames"
            if kind == "full":
                full_count += 1
            else:
                assert target_bbox is None, f"color {color} has multiple incomplete frames"
                target_bbox = component["bbox"]  # type: ignore[assignment]
                missing_side = kind

        assert full_count == 1, f"color {color} must have one complete frame"
        assert target_bbox is not None and missing_side is not None, f"color {color} needs one target frame"
        _fill_missing_side(out, color, target_bbox, missing_side)

    return out


def _sample_dims(rng: np.random.Generator) -> tuple[int, int]:
    h = int(rng.integers(MIN_FRAME_H, MAX_FRAME_H + 1))
    w = int(rng.integers(MIN_FRAME_W, MAX_FRAME_W + 1))
    return h, w


def _sample_pair_dims(rng: np.random.Generator) -> tuple[int, int, int, int]:
    src_h, src_w = _sample_dims(rng)
    dst_h, dst_w = _sample_dims(rng)
    while (src_h, src_w) == (dst_h, dst_w):
        dst_h, dst_w = _sample_dims(rng)
    return src_h, src_w, dst_h, dst_w


def _sample_missing_sides(rng: np.random.Generator, n: int) -> list[str]:
    sides: list[str] = []
    while len(sides) < n:
        batch = list(SIDES)
        rng.shuffle(batch)
        sides.extend(batch)
    return sides[:n]


def _choose_grid_shape(rng: np.random.Generator, n_colors: int) -> tuple[int, int]:
    h_lo, h_hi = GRID_H_BY_COLOR_COUNT[n_colors]
    w_lo, w_hi = GRID_W_BY_COLOR_COUNT[n_colors]
    h = int(rng.integers(h_lo, h_hi + 1))
    w = int(rng.integers(w_lo, w_hi + 1))
    return h, w


def _rect_is_free(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(occupied.shape[0], r0 + h + 1)
    cc1 = min(occupied.shape[1], c0 + w + 1)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _place_patches(
    rng: np.random.Generator,
    grid_shape: tuple[int, int],
    patches: list[np.ndarray],
) -> list[tuple[int, int]] | None:
    h, w = grid_shape
    occupied = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(patches)

    order = list(rng.permutation(len(patches)))
    order.sort(key=lambda idx: int(patches[idx].shape[0] * patches[idx].shape[1]), reverse=True)

    for idx in order:
        patch = patches[idx]
        ph, pw = map(int, patch.shape)
        if ph > h or pw > w:
            return None

        placed = False
        for _ in range(PLACE_ATTEMPTS):
            r0 = int(rng.integers(0, h - ph + 1))
            c0 = int(rng.integers(0, w - pw + 1))
            if not _rect_is_free(occupied, r0, c0, ph, pw):
                continue
            occupied[r0 : r0 + ph, c0 : c0 + pw] = True
            placements[idx] = (r0, c0)
            placed = True
            break
        if not placed:
            return None

    return [placement for placement in placements if placement is not None]


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    mask = patch != BG
    region = dst[r0 : r0 + h, c0 : c0 + w]
    region[mask] = patch[mask]


def _build_attempt(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    n_colors = int(rng.integers(3, 6))
    colors = [int(color) for color in rng.choice(np.arange(1, 10), size=n_colors, replace=False)]
    sides = _sample_missing_sides(rng, n_colors)

    objects: list[dict[str, object]] = []
    for color, side in zip(colors, sides):
        src_h, src_w, dst_h, dst_w = _sample_pair_dims(rng)
        objects.append(
            {
                "input": _full_frame_patch(src_h, src_w, color),
                "output": _full_frame_patch(src_h, src_w, color),
            }
        )
        objects.append(
            {
                "input": _missing_frame_patch(dst_h, dst_w, color, side),
                "output": _full_frame_patch(dst_h, dst_w, color),
            }
        )

    grid_shape = _choose_grid_shape(rng, n_colors)
    input_patches = [obj["input"] for obj in objects]  # type: ignore[index]
    placements = _place_patches(rng, grid_shape, input_patches)
    if placements is None:
        return None

    inp = np.full(grid_shape, BG, dtype=int)
    out = np.full(grid_shape, BG, dtype=int)
    for obj, (r0, c0) in zip(objects, placements):
        _paste(inp, obj["input"], r0, c0)  # type: ignore[index]
        _paste(out, obj["output"], r0, c0)  # type: ignore[index]

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    for _ in range(40):
        built = _build_attempt(_RNG)
        if built is None:
            continue
        inp, out = built
        try:
            if np.array_equal(main(inp), out):
                return inp
        except Exception:
            continue
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
