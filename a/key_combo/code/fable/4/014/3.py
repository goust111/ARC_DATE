from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
CLOUD = 8
SQUARE_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)

TEMPLATE_RETRIES = 300
NOISE_RETRIES = 120
PLACEMENT_RETRIES = 200
GENERATION_RETRIES = 300


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors4_count(mask: np.ndarray) -> np.ndarray:
    counts = np.zeros(mask.shape, dtype=int)
    counts[1:, :] += mask[:-1, :]
    counts[:-1, :] += mask[1:, :]
    counts[:, 1:] += mask[:, :-1]
    counts[:, :-1] += mask[:, 1:]
    return counts


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []

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
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _is_connected(mask: np.ndarray) -> bool:
    if not mask.any():
        return False
    return len(_components(mask)) == 1


def _template_is_valid(mask: np.ndarray) -> bool:
    if not mask.any() or mask.all():
        return False
    if not mask[0, :].any() or not mask[-1, :].any():
        return False
    if not mask[:, 0].any() or not mask[:, -1].any():
        return False
    if not _is_connected(mask):
        return False
    counts = _neighbors4_count(mask)
    return bool(np.all(counts[mask] >= 2))


def _apply_corner_notch(mask: np.ndarray, corner: str) -> None:
    k = mask.shape[0]
    if corner == "tl":
        cells = ((0, 0), (0, 1), (1, 0))
    elif corner == "tr":
        cells = ((0, k - 1), (0, k - 2), (1, k - 1))
    elif corner == "bl":
        cells = ((k - 1, 0), (k - 2, 0), (k - 1, 1))
    else:
        cells = ((k - 1, k - 1), (k - 2, k - 1), (k - 1, k - 2))
    for r, c in cells:
        mask[r, c] = False


def _inside_noise_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    counts = _neighbors4_count(mask)
    coords = np.argwhere(~mask)
    return [
        (int(r), int(c))
        for r, c in coords
        if int(counts[int(r), int(c)]) <= 1
    ]


def _make_template(k: int) -> np.ndarray:
    min_area = max(k * 2, int(np.ceil(k * k * 0.55)))
    max_removed = max(4, k * k // 3)

    for _ in range(TEMPLATE_RETRIES):
        mask = np.ones((k, k), dtype=bool)
        _apply_corner_notch(mask, random.choice(("tl", "tr", "bl", "br")))

        target_removed = random.randint(4, max_removed)
        attempts = 0
        while attempts < k * k * 12:
            attempts += 1
            if mask.size - int(mask.sum()) >= target_removed and random.random() < 0.4:
                break
            coords = np.argwhere(mask)
            rr, cc = map(int, coords[random.randrange(len(coords))])
            trial = mask.copy()
            trial[rr, cc] = False
            if int(trial.sum()) < min_area:
                continue
            if _template_is_valid(trial):
                mask = trial

        if not _template_is_valid(mask):
            continue
        if len(_inside_noise_candidates(mask)) == 0:
            continue
        return mask

    raise RuntimeError("failed to build a valid template")


def _make_square(k: int) -> np.ndarray:
    for _ in range(100):
        n_colors = random.randint(2, min(5, len(SQUARE_COLORS)))
        palette = random.sample(list(SQUARE_COLORS), n_colors)
        square = np.random.choice(palette, size=(k, k), replace=True).astype(int)
        if len(set(map(int, np.unique(square)))) >= 2:
            return square
    raise RuntimeError("failed to build a multicolor square")


def _nonadjacent(pos: tuple[int, int], chosen: list[tuple[int, int]]) -> bool:
    return all(abs(pos[0] - rr) + abs(pos[1] - cc) != 1 for rr, cc in chosen)


def _choose_noise_positions(template: np.ndarray, pad: int) -> list[tuple[int, int]] | None:
    k = template.shape[0]
    h = k + 2 * pad
    cloud = np.zeros((h, h), dtype=bool)
    cloud[pad : pad + k, pad : pad + k] = template
    counts = _neighbors4_count(cloud)

    inside = [
        (int(r) + pad, int(c) + pad)
        for r, c in np.argwhere(~template)
        if int(_neighbors4_count(template)[int(r), int(c)]) <= 1
    ]
    outside = [
        (int(r), int(c))
        for r, c in np.argwhere(np.ones_like(cloud, dtype=bool))
        if not (pad <= int(r) < pad + k and pad <= int(c) < pad + k)
        and int(counts[int(r), int(c)]) <= 1
    ]

    preferred_outside = [pos for pos in outside if int(counts[pos]) == 1]
    fallback_outside = [pos for pos in outside if int(counts[pos]) == 0]

    for _ in range(NOISE_RETRIES):
        chosen: list[tuple[int, int]] = []

        inside_order = inside[:]
        random.shuffle(inside_order)
        mandatory_inside = next((pos for pos in inside_order if _nonadjacent(pos, chosen)), None)
        if mandatory_inside is None:
            continue
        chosen.append(mandatory_inside)

        outside_order = preferred_outside[:]
        random.shuffle(outside_order)
        mandatory_outside = next((pos for pos in outside_order if _nonadjacent(pos, chosen)), None)
        if mandatory_outside is None:
            outside_order = fallback_outside[:]
            random.shuffle(outside_order)
            mandatory_outside = next((pos for pos in outside_order if _nonadjacent(pos, chosen)), None)
        if mandatory_outside is None:
            continue
        chosen.append(mandatory_outside)

        pool = [pos for pos in inside + outside if pos not in chosen]
        random.shuffle(pool)
        extra_target = random.randint(1, min(5, len(pool)))
        for pos in pool:
            if _nonadjacent(pos, chosen):
                chosen.append(pos)
                if len(chosen) >= 2 + extra_target:
                    break

        final_cloud = cloud.copy()
        for r, c in chosen:
            final_cloud[r, c] = True
        final_counts = _neighbors4_count(final_cloud)
        if all(int(final_counts[r, c]) <= 1 for r, c in chosen):
            return chosen

    return None


def _build_cloud_patch(template: np.ndarray, pad: int) -> np.ndarray:
    k = template.shape[0]
    patch = np.zeros((k + 2 * pad, k + 2 * pad), dtype=int)
    patch[pad : pad + k, pad : pad + k][template] = CLOUD
    noise_positions = _choose_noise_positions(template, pad)
    if noise_positions is None:
        raise RuntimeError("failed to add valid noise")
    for r, c in noise_positions:
        patch[r, c] = CLOUD
    return patch


def _separated(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _place_objects(cloud_patch: np.ndarray, square: np.ndarray) -> np.ndarray | None:
    ch, cw = cloud_patch.shape
    k = square.shape[0]
    min_h = max(ch, k) + 3
    min_w = max(cw, k) + 3

    for _ in range(PLACEMENT_RETRIES):
        h = random.randint(min_h, 30)
        w = random.randint(min_w, 30)
        cr = random.randint(0, h - ch)
        cc = random.randint(0, w - cw)
        sr = random.randint(0, h - k)
        sc = random.randint(0, w - k)
        cloud_box = (cr, cc, cr + ch - 1, cc + cw - 1)
        square_box = (sr, sc, sr + k - 1, sc + k - 1)
        if not _separated(cloud_box, square_box, gap=2):
            continue

        grid = np.zeros((h, w), dtype=int)
        grid[cr : cr + ch, cc : cc + cw] = cloud_patch
        grid[sr : sr + k, sc : sc + k] = square
        return grid
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden stencil from the noisy cyan cloud and use it to crop the multicolor square.

    Input structure:
    - The grid has a black background and exactly one dense k x k multicolor square.
    - Elsewhere, cyan pixels form a noisy cloud whose supported core is the true stencil.
    - The stencil's supported pixels are 4-connected, and each real stencil pixel has at least two cyan 4-neighbors.

    Rule:
    - Find the dense multicolor square and crop it to its tight k x k bbox.
    - Treat every other non-black pixel as part of the cloud, then keep only cloud pixels with at least two cloud 4-neighbors.
    - Crop the supported cloud to its tight k x k bbox and use that binary mask on the square.

    Key insight:
    - The visible cyan shape is not the stencil itself: unsupported cyan pixels are noise, so the answer appears only after denoising by local 4-neighbor support.
    """
    g = np.asarray(grid, dtype=int)
    foreground = g != BLACK

    square_candidates: list[tuple[int, int, int, int, np.ndarray]] = []
    for coords in _components(foreground):
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h != w or int(coords.shape[0]) != h * w:
            continue
        colors = {int(g[int(r), int(c)]) for r, c in coords}
        if len(colors) < 2:
            continue
        square_candidates.append((h * w, int(r0), int(c0), int(r1), coords))

    assert square_candidates, "expected a dense multicolor square"
    _, r0, c0, r1, square_coords = max(square_candidates, key=lambda item: item[0])
    k = int(r1 - r0 + 1)
    square_sprite = g[r0 : r0 + k, c0 : c0 + k].copy()

    square_mask = np.zeros_like(foreground, dtype=bool)
    square_mask[square_coords[:, 0], square_coords[:, 1]] = True
    cloud_mask = foreground & ~square_mask

    supported = cloud_mask & (_neighbors4_count(cloud_mask) >= 2)
    box = _bbox(supported)
    assert box is not None, "expected a supported stencil after denoising"
    stencil = _crop(supported.astype(bool), box)
    assert stencil.shape == square_sprite.shape, "stencil bbox must match the square size"

    out = square_sprite.copy()
    out[~stencil] = BLACK
    return out


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        k = random.randint(4, 8)
        template = _make_template(k)
        square = _make_square(k)
        pad = random.randint(2, 4)

        try:
            cloud_patch = _build_cloud_patch(template, pad)
        except RuntimeError:
            continue

        grid = _place_objects(cloud_patch, square)
        if grid is None:
            continue

        expected = square.copy()
        expected[~template] = BLACK

        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid

    raise RuntimeError("failed to generate a valid input")
