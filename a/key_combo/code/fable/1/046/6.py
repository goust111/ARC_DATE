import itertools
import random

import numpy as np


BLACK = 0
MASK_COLOR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
TRANSFORMS = (
    (False, False),  # identity
    (True, False),   # left-right mirror
    (False, True),   # invert black/foreground
    (True, True),    # mirror + invert
)

GEN_ATTEMPTS = 400
BASE_ATTEMPTS = 200
MASK_ATTEMPTS = 200
PANEL_H_RANGE = (5, 14)
PANEL_W_RANGE = (5, 14)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Inclusive bbox copied from arc_utils' (r0, c0, r1, c1) convention."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _apply_transform(panel: np.ndarray, fg: int, mirror: bool, invert: bool) -> np.ndarray:
    out = np.fliplr(panel) if mirror else panel.copy()
    if invert:
        out = np.where(out == BLACK, fg, BLACK)
    return out


def _panel_slices(panel_h: int, panel_w: int) -> list[tuple[slice, slice]]:
    return [
        (slice(0, panel_h), slice(0, panel_w)),
        (slice(0, panel_h), slice(panel_w + 1, 2 * panel_w + 1)),
        (slice(panel_h + 1, 2 * panel_h + 1), slice(0, panel_w)),
        (slice(panel_h + 1, 2 * panel_h + 1), slice(panel_w + 1, 2 * panel_w + 1)),
    ]


def _extract_panels(grid: np.ndarray) -> tuple[list[np.ndarray], int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h % 2 == 1 and w % 2 == 1, "separator row/column require odd grid dimensions"
    panel_h, panel_w = h // 2, w // 2
    sep_r, sep_c = panel_h, panel_w
    assert np.all(g[sep_r, :] == g[sep_r, 0]), "separator row must be solid"
    assert np.all(g[:, sep_c] == g[0, sep_c]), "separator column must be solid"
    panels = [g[rs, cs].copy() for rs, cs in _panel_slices(panel_h, panel_w)]
    return panels, panel_h, panel_w


def _infer_fg_and_mask_color(panels: list[np.ndarray]) -> tuple[int, int]:
    support: dict[int, int] = {}
    for panel in panels:
        for color in np.unique(panel):
            value = int(color)
            if value == BLACK:
                continue
            support[value] = support.get(value, 0) + 1
    assert support, "expected at least one non-black panel color"
    fg = max(support, key=lambda color: support[color])
    extras = [color for color in support if color != fg]
    assert len(extras) == 1, "expected exactly one masking color in the panel area"
    return int(fg), int(extras[0])


def _distinct_predictions(panels: list[np.ndarray], fg: int, mask_color: int) -> tuple[int, list[np.ndarray]]:
    masked_indices = [idx for idx, panel in enumerate(panels) if np.any(panel == mask_color)]
    assert len(masked_indices) == 1, "expected exactly one masked panel"
    masked_idx = masked_indices[0]
    visible = panels[masked_idx] != mask_color
    complete_idx = next(idx for idx in range(4) if idx != masked_idx)

    predictions: list[np.ndarray] = []
    for perm in itertools.permutations(range(4)):
        mirror, invert = TRANSFORMS[perm[complete_idx]]
        base = _apply_transform(panels[complete_idx], fg, mirror, invert)
        candidate = None
        ok = True
        for idx, panel in enumerate(panels):
            exp_mirror, exp_invert = TRANSFORMS[perm[idx]]
            expected = _apply_transform(base, fg, exp_mirror, exp_invert)
            if idx == masked_idx:
                if not np.array_equal(expected[visible], panel[visible]):
                    ok = False
                    break
                candidate = expected
            elif not np.array_equal(expected, panel):
                ok = False
                break
        if ok:
            assert candidate is not None
            predictions.append(candidate)

    unique_predictions: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for pred in predictions:
        key = (tuple(map(int, pred.shape)), pred.tobytes())
        if key not in seen:
            seen.add(key)
            unique_predictions.append(pred)
    return masked_idx, unique_predictions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the gray-covered panel by matching the four mirror/invert variants of one binary template.

    Input structure:
    - A solid separator row and a solid separator column split the grid into four equal panels.
    - Each panel uses only black and one shared foreground color, except one panel also contains a gray solid rectangular mask.
    - The four panels are the original binary pattern, its left-right mirror, its foreground/background inversion, and the mirrored inversion, in unknown order.

    Rule:
    - Extract the four panels and identify the shared foreground color plus the single masking color.
    - Try every one-to-one assignment of the four mirror/invert labels to the four panel positions, keeping only assignments that exactly match every complete panel and the visible cells of the masked panel.
    - Fill the gray rectangle with the consensus colors predicted for the masked panel and leave every other cell unchanged.

    Key insight:
    - The missing region cannot be recovered from symmetry alone because two panels are complemented; the solver must jointly infer both the mirror bit and the invert bit before copying values into the mask.
    """
    g = np.asarray(grid, dtype=int)
    panels, panel_h, panel_w = _extract_panels(g)
    fg, mask_color = _infer_fg_and_mask_color(panels)
    masked_idx, predictions = _distinct_predictions(panels, fg, mask_color)
    assert predictions, "no consistent reconstruction found"
    assert len(predictions) == 1, "masked panel reconstruction must be unique"

    mask = panels[masked_idx] == mask_color
    box = _bbox(mask)
    assert box is not None, "mask must be present"
    r0, c0, r1, c1 = box
    assert np.all(mask[r0 : r1 + 1, c0 : c1 + 1]), "mask must be a solid rectangle"
    assert int(mask.sum()) == (r1 - r0 + 1) * (c1 - c0 + 1), "mask rectangle must have no holes"

    out = g.copy()
    panel_view = out[_panel_slices(panel_h, panel_w)[masked_idx]]
    panel_view[mask] = predictions[0][mask]
    return out


def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def _add_rect(mask: np.ndarray, value: bool) -> None:
    h, w = mask.shape
    rh = random.randint(1, max(1, h // 2))
    rw = random.randint(1, max(1, w // 2))
    r0 = random.randint(0, h - rh)
    c0 = random.randint(0, w - rw)
    mask[r0 : r0 + rh, c0 : c0 + rw] = value


def _add_line(mask: np.ndarray, value: bool) -> None:
    h, w = mask.shape
    if random.random() < 0.5:
        r = random.randrange(h)
        c0 = random.randint(0, w - 2)
        c1 = random.randint(c0 + 1, w - 1)
        mask[r, c0 : c1 + 1] = value
    else:
        c = random.randrange(w)
        r0 = random.randint(0, h - 2)
        r1 = random.randint(r0 + 1, h - 1)
        mask[r0 : r1 + 1, c] = value


def _add_l(mask: np.ndarray, value: bool) -> None:
    h, w = mask.shape
    r0 = random.randint(0, h - 2)
    c0 = random.randint(0, w - 2)
    r1 = random.randint(r0 + 1, h - 1)
    c1 = random.randint(c0 + 1, w - 1)
    if random.random() < 0.5:
        mask[r0 : r1 + 1, c0] = value
        mask[r1, c0 : c1 + 1] = value
    else:
        mask[r0 : r1 + 1, c1] = value
        mask[r0, c0 : c1 + 1] = value


def _enforce_asymmetry(mask: np.ndarray) -> None:
    h, w = mask.shape
    pairs = [(r, c, w - 1 - c) for r in range(h) for c in range(w // 2)]
    if not pairs:
        return

    if not any(mask[r, c] != mask[r, mc] for r, c, mc in pairs):
        r, c, mc = random.choice(pairs)
        mask[r, c] = True
        mask[r, mc] = False

    if not any(mask[r, c] == mask[r, mc] for r, c, mc in pairs):
        r, c, mc = random.choice(pairs)
        value = random.random() < 0.5
        mask[r, c] = value
        mask[r, mc] = value


def _generate_base_mask(panel_h: int, panel_w: int) -> np.ndarray:
    for _ in range(BASE_ATTEMPTS):
        mask = np.zeros((panel_h, panel_w), dtype=bool)

        for _ in range(random.randint(2, 4)):
            random.choice((_add_rect, _add_rect, _add_line, _add_l))(mask, True)

        for _ in range(random.randint(0, 2)):
            random.choice((_add_rect, _add_line))(mask, False)

        for _ in range(random.randint(1, 2)):
            mask[random.randrange(panel_h), random.randrange(panel_w)] = True

        _enforce_asymmetry(mask)

        density = float(mask.mean())
        if not (0.22 <= density <= 0.68):
            continue
        if not mask.any() or mask.all():
            continue

        mirror = np.fliplr(mask)
        anti = np.logical_not(mirror)
        if np.array_equal(mask, mirror):
            continue
        if np.array_equal(mask, anti):
            continue
        area = panel_h * panel_w
        if _hamming(mask, mirror) < max(4, area // 7):
            continue
        if _hamming(mask, anti) < max(4, area // 7):
            continue
        return mask

    raise RuntimeError("failed to generate a distinctive base pattern")


def _mask_rectangle_candidates(panel_h: int, panel_w: int) -> list[tuple[int, int, int, int]]:
    candidates: list[tuple[int, int, int, int]] = []
    max_area = max(4, (panel_h * panel_w) // 3)
    for mask_h in range(1, panel_h + 1):
        for mask_w in range(1, panel_w + 1):
            area = mask_h * mask_w
            if area < 2 or area > max_area:
                continue
            if area >= panel_h * panel_w:
                continue
            for r0 in range(panel_h - mask_h + 1):
                for c0 in range(panel_w - mask_w + 1):
                    candidates.append((r0, c0, mask_h, mask_w))
    random.shuffle(candidates)
    return candidates


def _choose_mask_rectangle(full_panels: list[np.ndarray], masked_idx: int, fg: int) -> tuple[int, int, int, int] | None:
    target = full_panels[masked_idx]
    tried = 0
    for r0, c0, mask_h, mask_w in _mask_rectangle_candidates(*target.shape):
        region = target[r0 : r0 + mask_h, c0 : c0 + mask_w]
        if len(np.unique(region)) < 2:
            continue
        masked_panels = [panel.copy() for panel in full_panels]
        masked_panels[masked_idx][r0 : r0 + mask_h, c0 : c0 + mask_w] = MASK_COLOR
        _, predictions = _distinct_predictions(masked_panels, fg, MASK_COLOR)
        if len(predictions) == 1 and np.array_equal(predictions[0], target):
            return r0, c0, mask_h, mask_w
        tried += 1
        if tried >= MASK_ATTEMPTS:
            break
    return None


def _assemble_grid(panels: list[np.ndarray], separator_color: int) -> np.ndarray:
    panel_h, panel_w = map(int, panels[0].shape)
    grid = np.full((2 * panel_h + 1, 2 * panel_w + 1), separator_color, dtype=int)
    for panel, (rs, cs) in zip(panels, _panel_slices(panel_h, panel_w), strict=True):
        grid[rs, cs] = panel
    return grid


def _generate_example() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(GEN_ATTEMPTS):
        panel_h = random.randint(*PANEL_H_RANGE)
        panel_w = random.randint(*PANEL_W_RANGE)
        fg = random.choice(PALETTE)
        separator_color = random.choice([color for color in PALETTE if color != fg])

        base_mask = _generate_base_mask(panel_h, panel_w)
        base_panel = np.where(base_mask, fg, BLACK).astype(int)

        transform_order = random.sample(range(4), 4)
        full_panels = [
            _apply_transform(base_panel, fg, *TRANSFORMS[transform_idx]) for transform_idx in transform_order
        ]

        masked_idx = random.randrange(4)
        mask_box = _choose_mask_rectangle(full_panels, masked_idx, fg)
        if mask_box is None:
            continue

        r0, c0, mask_h, mask_w = mask_box
        input_panels = [panel.copy() for panel in full_panels]
        input_panels[masked_idx][r0 : r0 + mask_h, c0 : c0 + mask_w] = MASK_COLOR

        inp = _assemble_grid(input_panels, separator_color)
        out = _assemble_grid(full_panels, separator_color)

        try:
            pred = main(inp)
        except AssertionError:
            continue
        if not np.array_equal(pred, out):
            continue
        return inp, out

    raise RuntimeError("failed to generate a valid puzzle instance")


def generate_input() -> np.ndarray:
    return _generate_example()[0]
